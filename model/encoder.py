import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, vocab_size=20000, embedding_size=512, hidden_size=1024, lstm_layers=1, latent_dim=32, dropout=.3):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(input_size=embedding_size,
                           hidden_size=hidden_size,
                           num_layers=lstm_layers,
                           batch_first=True, dropout=dropout)
        self.mu = nn.Linear(hidden_size, latent_dim)
        self.logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, input, lens):#, final_inds):
        """
        :param input: [batch_size, max_seq_len] tensor
        :param lens: [batch_size, max_seq_len] tensor witih all 0's and a 1 in the place of the last word
        :return:]
        """
        embeddings = self.embedding(input)
        packed_input = pack_padded_sequence(embeddings, lens, batch_first=True)
        _, (h_n, _) = self.rnn(packed_input)
        rnn_final_hiddens = h_n[0, :, :]
        mu = self.mu(rnn_final_hiddens)
        logvar = self.logvar(rnn_final_hiddens)
        return mu, logvar, embeddings

    def sample_z(self, mu, log_var):
        std = (log_var / 2.0).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add(mu)


if __name__ == "__main__":
    from data.dataset import TextDataset, pad_batch
    from torch.utils.data import DataLoader
    test_ds = TextDataset("../data/yelp_data/vocab.txt", "../data/yelp_data/part_0")
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=True, num_workers=1, collate_fn=pad_batch)
    enc = Encoder()
    for batch, lens in test_loader:
        m, lv, em = enc(Variable(batch), lens)
        z = enc.sample_z(m, lv)
        break
