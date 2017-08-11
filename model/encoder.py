import torch.nn as nn
import torch.nn.functional as F
import torch


class Encoder(nn.Module):
    def __init__(self, vocab_size=20000, embedding_size=512, hidden_size=1024, lstm_layers=1, latent_dim=32):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(input_size=embedding_size,
                           hidden_size=hidden_size,
                           num_layers=lstm_layers,
                           batch_first=True)
        self.mu = nn.Linear(hidden_size, latent_dim)
        self.logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, input, final_inds):
        """
        :param input: [batch_size, max_seq_len] tensor
        :param lens: [batch_size] tensor
        :return:]
        """

        embeddings = self.embedding(input)
        rnn_hidden, (_, _) = self.rnn(embeddings)
        # get final embeddings
        # expand final_inds
        exp_final_inds = final_inds.view(final_inds.size(0), final_inds.size(1), 1).repeat(1, 1, rnn_hidden.size(2))
        # mask out all but final hidden states
        rnn_final_hiddens = rnn_hidden.mul(exp_final_inds.float()).sum(dim=1)[:, 0, :]
        mu = self.mu(rnn_final_hiddens)
        logvar = self.logvar(rnn_final_hiddens)

        return mu, logvar


from data.dataset import TextDataset, pad_batch
from torch.utils.data import DataLoader
from torch.autograd import Variable
test_ds = TextDataset("../data/yelp_data/vocab.txt", "../data/yelp_data/part_0")
test_loader = DataLoader(test_ds, batch_size=8, shuffle=True, num_workers=4, collate_fn=pad_batch)
enc = Encoder()
for batch, lens in test_loader:
    out = enc(Variable(batch), Variable(lens))
    break
