from __future__ import print_function
import argparse
from model.rvae import RVAE
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os
import numpy as np
from tensorboard_logger import configure, log_value
from data.dataset import TextDataset, pad_batch
from torch.utils.data import DataLoader
import time

parser = argparse.ArgumentParser(description='test vae')
parser.add_argument('--embedding-size', type=int, default=512, metavar='EMB',
                    help='size of word embeddings (default: 512)')
parser.add_argument('--vocab-size', type=int, default=20000, metavar='EMB',
                    help='number of words in vocab (default: 20k)')
parser.add_argument('--kl-iter', type=int, default=10000, metavar='EMB',
                    help='number of iters to anneal the kl term (default: 10k)')
parser.add_argument('--hidden-size', type=int, default=1024, metavar='HDS',
                    help='size of LSTM hidden layer (default: 1024)')
parser.add_argument('--lstm-layers', type=int, default=1, metavar='RLS',
                    help='number of layers in LSTM (default: 1)')
parser.add_argument('--latent-dim', type=int, default=32, metavar='LTD',
                    help='dimensionality of latent state (default: 32)')
parser.add_argument('--layer-list', type=str, default="1,2,4", metavar='LL',
                    help='list of ints telling the size of the decoder (default: "1,2,4")')
parser.add_argument('--dropout', type=float, default=.3, metavar='DRO',
                    help='dropout probability for encoder LSTM (default: .3')
parser.add_argument('--train-dir', type=str, default="/tmp/rvae_train", metavar='TD',
                    help='where to save logs and checkpointed models')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 40)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')
parser.add_argument('--sample-interval', type=int, default=100, metavar='LI',
                    help='how many batches to wait before sampling reconstructions')
parser.add_argument('--test-interval', type=int, default=1, metavar='TEI',
                    help='how many epochs to wait before running testing')
parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                    help='size of word embeddings (default: 512)')


def lr_scheduler(opt, epoch, init_lr=0.001, start_decay_epoch=30, decay_multiplier=.5):
    """Decay learning rate by a factor of 0.5 every lr_decay_epoch epochs."""
    if epoch < start_decay_epoch:
        lr = init_lr
    else:
        num_decays = (epoch - (start_decay_epoch - 1)) // 2
        lr = init_lr * (decay_multiplier**num_decays)
        print('Learning Rate is set to {}'.format(lr))

    for param_group in opt.param_groups:
        param_group['lr'] = lr

    return opt


def int2sent(seq, vocab):
    """

    :param seq: numpy array 1D
    :param vocab: dict {ind: word}
    :return: string of the sentance represented in seq
    """
    s = []
    for idx in seq:
        cur = vocab[idx]
        if cur == '<eos>':
            break
        s.append(cur)
    return ' '.join(s)


def sample_reconst(data, output, vocab):
    out = []
    for x, xp in zip(data, output):
        sx = int2sent(x, vocab)
        sxp = int2sent(xp, vocab)
        out.append((sx, sxp))
    return out


def train(args, epoch, optimizer, model):
    optimizer = lr_scheduler(optimizer, epoch)
    train_dataset = TextDataset("data/yelp_data/vocab.txt", "data/yelp_data/part_0")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, collate_fn=pad_batch
    )
    for idx, batch in enumerate(train_loader):
        step = len(train_loader) * epoch + idx
        start_time = time.time()
        data, lens, mask = batch
        data = Variable(data)
        lens = Variable(lens)
        mask = Variable(mask)
        if args.cuda:
            data = data.cuda()
            lens = lens.cuda()
            mask = mask.cuda()

        optimizer.zero_grad()
        out, nll, kld = model(data, lens, mask)
        kl_weight = min((float(step) / args.kl_iter) * .99 + .01, 1.0)
        weighted_kld = kld.mul(kl_weight)
        loss = nll + weighted_kld
        loss.backward()
        optimizer.step()
        batch_time = time.time() - start_time

        if idx % args.log_interval == 0:
            l, n, k = loss.data[0], nll.data[0], kld.data[0]
            log_value("total_loss", l, step)
            log_value("nll", n, step)
            log_value("kld", k, step)
            print("Total Loss: {}, NLL: {}, KLD: {} ({} sec/batch)".format(l, n, k, batch_time))

        if idx % args.sample_interval == 0:
            data_np = data.data.cpu().numpy()
            _, out_preds = torch.max(out, 1)
            out_preds_np = out_preds[:, 0, :].data.cpu().numpy()
            recons = sample_reconst(data_np, out_preds_np, train_dataset.ind2word)
            for i in range(1000):
                print(i, train_dataset.ind2word[i])
            with open(os.path.join(args.train_dir, "reconstructions_{}.txt".format(idx)), "w") as f:
                for s, sp in recons:
                    f.write(s + '\n')
                    f.write(sp + '\n')
                    f.write('\n')




if __name__ == "__main__":
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.layer_list = [int(c) for c in args.layer_list.split(',')]
    # create training directory and delete if already exists
    if os.path.exists(args.train_dir):
        print("Deleting existing train dir")
        import shutil
        shutil.rmtree(args.train_dir)

    os.makedirs(args.train_dir)
    configure(args.train_dir, flush_secs=5)

    model = RVAE(
        args.vocab_size, args.embedding_size, args.hidden_size,
        args.lstm_layers, args.latent_dim, args.layer_list, args.dropout
    )
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), betas=(.5, .999))

    for epoch in range(args.epochs):
        train(args, epoch, optimizer, model)
        #test(args, epoch, model)