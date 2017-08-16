import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from decoder import CNNDecoder
from encoder import Encoder


class RVAE(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, lstm_layers, latent_dim, layers_list=[1,2,4], dropout=.3):
        super(RVAE, self).__init__()

        self.encoder = Encoder(
            vocab_size=vocab_size, embedding_size=embedding_size,
            hidden_size=hidden_size, lstm_layers=lstm_layers,
            latent_dim=latent_dim, dropout=dropout
        )
        self.decoder = CNNDecoder(
            embedding_size, latent_dim,
            layers_list=layers_list, vocab_size=vocab_size
        )

    def forward(self, data, lens):
        """

        :param data: [batch_size, seq_length] tensor containing word indices
        :param lens: list of ints of sequence lengths
        :return:
        """

        mu, logvar, embeddings = self.encoder(data, lens)
        z = self.encoder.sample_z(mu, logvar)
        kld = (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 1)).mean().squeeze()
        out = self.decoder(embeddings, z)
        # size is [batch_size, vocab_size, seq_length], need to flip to [batch_size, seq_length, vocab_size]
        out_perm = out.permute(0, 2, 1).contiguous()
        # squash to [batch_size * seq_length, vocab_size]
        out_exp = out_perm.view(out.size(0) * out.size(2), out.size(1))
        log_probs_exp = F.log_softmax(out_exp)
        if torch.cuda.is_available():
            data = data.cuda()
        data_exp = data.view((-1, 1))
        nll_exp = torch.gather(log_probs_exp, 1, data_exp)
        nll = nll_exp.view(out.size(0), out.size(2))
        # the above should be of size [batch size, sequence length]
        # create mask for loss
        mask = np.zeros((nll.size()), dtype=np.float)
        for i, l in enumerate(lens):
            mask[i, :l] = 1.0
        mask = Variable(torch.FloatTensor(mask))
        if torch.cuda.is_available():
            mask = mask.cuda()
        masked_nll = nll.mul(mask).sum(dim=1).mean().squeeze().mul(-1.)
        return out, masked_nll, kld

    # def trainer(self, optimizer, batch_loader):
    #     def train(i, batch_size, use_cuda, dropout):
    #         input = batch_loader.next_batch(batch_size, 'train')
    #         input = [Variable(t.from_numpy(var)) for var in input]
    #         input = [var.long() for var in input]
    #         input = [var.cuda() if use_cuda else var for var in input]
    #
    #         [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input
    #
    #         logits, _, kld = self(dropout,
    #                               encoder_word_input, encoder_character_input,
    #                               decoder_word_input, decoder_character_input,
    #                               z=None)
    #
    #         logits = logits.view(-1, self.params.word_vocab_size)
    #         target = target.view(-1)
    #         cross_entropy = F.cross_entropy(logits, target)
    #
    #         loss = 79 * cross_entropy + kld_coef(i) * kld
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         return cross_entropy, kld, kld_coef(i)
    #
    #     return train
    #
    # def validater(self, batch_loader):
    #     def validate(batch_size, use_cuda):
    #         input = batch_loader.next_batch(batch_size, 'valid')
    #         input = [Variable(t.from_numpy(var)) for var in input]
    #         input = [var.long() for var in input]
    #         input = [var.cuda() if use_cuda else var for var in input]
    #
    #         [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input
    #
    #         logits, _, kld = self(0.,
    #                               encoder_word_input, encoder_character_input,
    #                               decoder_word_input, decoder_character_input,
    #                               z=None)
    #
    #         logits = logits.view(-1, self.params.word_vocab_size)
    #         target = target.view(-1)
    #
    #         cross_entropy = F.cross_entropy(logits, target)
    #
    #         return cross_entropy, kld
    #
    #     return validate
    #
    # def sample(self, batch_loader, seq_len, seed, use_cuda):
    #     seed = Variable(t.from_numpy(seed).float())
    #     if use_cuda:
    #         seed = seed.cuda()
    #
    #     decoder_word_input_np, decoder_character_input_np = batch_loader.go_input(1)
    #
    #     decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
    #     decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())
    #
    #     if use_cuda:
    #         decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()
    #
    #     result = ''
    #
    #     initial_state = None
    #
    #     for i in range(seq_len):
    #         logits, initial_state, _ = self(0., None, None,
    #                                         decoder_word_input, decoder_character_input,
    #                                         seed, initial_state)
    #
    #         logits = logits.view(-1, self.params.word_vocab_size)
    #         prediction = F.softmax(logits)
    #
    #         word = batch_loader.sample_word_from_distribution(prediction.data.cpu().numpy()[-1])
    #
    #         if word == batch_loader.end_token:
    #             break
    #
    #         result += ' ' + word
    #
    #         decoder_word_input_np = np.array([[batch_loader.word_to_idx[word]]])
    #         decoder_character_input_np = np.array([[batch_loader.encode_characters(word)]])
    #
    #         decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
    #         decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())
    #
    #         if use_cuda:
    #             decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()
    #
    #     return result


if __name__ == "__main__":
    from data.dataset import TextDataset, pad_batch
    from torch.utils.data import DataLoader
    test_ds = TextDataset("../data/yelp_data/vocab.txt", "../data/yelp_data/part_0")
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=True, num_workers=4, collate_fn=pad_batch)
    vae = RVAE(20000, 512, 1024, 1, 32)
    d = vae.state_dict()
    for k, v in d.items():
        print(k, v.size())
    for batch, lens in test_loader:
        #print(lens[:2])
        #print(mask[:2])
        pred, nll, kld = vae(Variable(batch), lens)
        break

