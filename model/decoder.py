import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.functional import parameters_allocation_check


class RNNDecoder(nn.Module):
    def __init__(self, params):
        super(RNNDecoder, self).__init__()

        self.params = params

        self.rnn = nn.LSTM(input_size=self.params.latent_variable_size + self.params.word_embed_size,
                           hidden_size=self.params.decoder_rnn_size,
                           num_layers=self.params.decoder_num_layers,
                           batch_first=True)

        self.fc = nn.Linear(self.params.decoder_rnn_size, self.params.word_vocab_size)

    def forward(self, decoder_input, z, drop_prob, initial_state=None):
        """
        :param decoder_input: tensor with shape of [batch_size, seq_len, embed_size]
        :param z: sequence context with shape of [batch_size, latent_variable_size]
        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout
        :param initial_state: initial state of decoder rnn

        :return: unnormalized logits of sentense words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'

        [batch_size, seq_len, _] = decoder_input.size()

        '''
            decoder rnn is conditioned on context via additional bias = W_cond * z to every input token
        '''
        decoder_input = F.dropout(decoder_input, drop_prob)

        z = t.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.latent_variable_size)
        decoder_input = t.cat([decoder_input, z], 2)

        rnn_out, final_state = self.rnn(decoder_input, initial_state)

        rnn_out = rnn_out.contiguous().view(-1, self.params.decoder_rnn_size)
        result = self.fc(rnn_out)
        result = result.view(batch_size, seq_len, self.params.word_vocab_size)

        return result, final_state


class CausalConv1D(nn.Module):
    """
    Assumes that the input padded as [START, word1, word2, word3, ..., wordN] len = N + 1
    Outputs will be preds for [word1, word2, ..., wordN, EOS] len = N + 1
    """
    def __init__(self, D_in, D_out, k_size, dilation):
        r_field = k_size + (k_size - 1) * (dilation - 1)
        padding = r_field - 1
        self.conv = torch.nn.Conv1d(D_in, D_out, k_size, dilation=dilation, padding=padding)

    def forward(self, x):
        # the output of the convolution will have trailing values that we don't want
        x_conv_padded = self.conv(x)
        x_conv = x_conv_padded[:, :, :x.size(2)]
        return x_conv


class ARBlock(nn.Module):
    def __init__(self, D_in, k_size, dilation, nonlin=torch.nn.ReLU,
                 internal_size=512, external_size=1024):
        self.c1 = CausalConv1D(D_in, internal_size, 1, 1)
        self.c2 = CausalConv1D(512, internal_size, k_size, dilation)
        self.c3 = CausalConv1D(512, external_size, 1, 1)

        # if D_in and external_size do not match, we use a linear [1x1] conv (no nonlinearity)
        # to project the input to the right dimensionality
        if D_in != external_size:
            self.need_proj = True
            self.c_proj = CausalConv1D(D_in, external_size, 1, 1)
        else:
            self.need_proj = False

        self.nonlin = nonlin()

    def forward(self, x):
        xc1 = self.nonlin(self.c1(x))
        xc2 = self.nonlin(self.c2(xc1))
        xc3 = self.nonlin(self.c3(xc2))
        if self.need_proj:
            x_proj = self.c_proj(x)
            return xc3.add(x_proj)
        else:
            return xc3.add(x)


class CNNDecoder(nn.Module):
    def __init__(self, embed_dim, z_dim, layers_list=[1,2,4], internal_size=512, external_size=1024, k_size=3):
        """

        :param layers_list: list in form of [1, 2, 4] that specifies 3 residual blocks
            with dilations of 1, 2, and 4
        :param internal_size:
        :param external_size:
        """
        super(CNNDecoder, self).__init__()
        self.blocks = []
        for i, dilation in enumerate(layers_list):
            self.blocks.append(
                ARBlock(
                    embed_dim + z_dim if i == 0 else external_size,
                    k_size, dilation,
                    internal_size=internal_size, external_size=external_size
                )
            )

    def forward(self, x, z):
        """

        :param x: sentence tensor [batch_size, embed_dim, num_words]
        :param z: context tensor [batch_size, z_dim]
        :return:
        """
        # Add a single 0 example onto x
        x_dim_exp = x.view(x.size(0), x.size(1), x.size(2), 1).permute(0, 3, 1, 2)
        # CHECK SIZES HERE MAKE SURE THIS IS RIGHT!!
        # need to swap the axis cuz this operation only works on images
        x_dim_exp_padded = F.pad(x_dim_exp, (0, 0, 1, 0))
        x_padded = x_dim_exp_padded[:, 0, :, :]
        # need to copy z num_words + 1 times in the 2nd dimension then concat to x_padded
        z_exp = z.view(z.size(0), z.size(1), 1).expand((z.size(0), z.size(1), x.size(2) + 1))
        dec_input = torch.cat([x_padded, z_exp], 1)

        x_cur = dec_input
        for block in self.blocks:
            x_cur = block(x_cur)

        return x_cur



