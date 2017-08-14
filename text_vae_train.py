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
import time

parser = argparse.ArgumentParser(description='test vae')
parser.add_argument('--embedding-size', type=int, default=512, metavar='EMB',
                    help='size of word embeddings (default: 512)')
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
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test-interval', type=int, default=1, metavar='TEI',
                    help='how many epochs to wait before running testing')


if __name__ == "__main__":
    args = parser.parse_args()
    args.layer_list = [int(c) for c in args.layer_list]
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