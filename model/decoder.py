import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CausalConv1D(nn.Module):
    """
    Assumes that the input padded as [START, word1, word2, word3, ..., wordN, EOS] len = N + 2
    Outputs will be preds for [word1, word2, ..., wordN, EOS] len = N + 1
    """
    def __init__(self, D_in, D_out, k_size, dilation, do=.1):
        super(CausalConv1D, self).__init__()
        r_field = k_size + (k_size - 1) * (dilation - 1)
        padding = r_field - 1
        self.conv = torch.nn.Conv1d(D_in, D_out, k_size, dilation=dilation, padding=padding)
        self.dropout = nn.Dropout(p=do)

    def forward(self, x):
        # the output of the convolution will have trailing values that we don't want
        x_conv_padded = self.conv(self.dropout(x))
        x_conv = x_conv_padded[:, :, :x.size(2)]
        return x_conv


class ARBlock(nn.Module):
    def __init__(self, D_in, k_size, dilation, nonlin=torch.nn.ReLU,
                 internal_size=512, external_size=1024, do=.1):
        super(ARBlock, self).__init__()
        self.c1 = CausalConv1D(D_in, internal_size, 1, 1, do=do)
        self.c2 = CausalConv1D(512, internal_size, k_size, dilation, do=do)
        self.c3 = CausalConv1D(512, external_size, 1, 1, do=do)

        # if D_in and external_size do not match, we use a linear [1x1] conv (no nonlinearity)
        # to project the input to the right dimensionality
        if D_in != external_size:
            self.need_proj = True
            self.c_proj = CausalConv1D(D_in, external_size, 1, 1, do=do)
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
    def __init__(self, embed_dim, z_dim, layers_list=[1,2,4], internal_size=512, external_size=1024, k_size=3, vocab_size=20000, do=.1):
        """

        :param layers_list: list in form of [1, 2, 4] that specifies 3 residual blocks
            with dilations of 1, 2, and 4
        :param internal_size:
        :param external_size:
        """
        super(CNNDecoder, self).__init__()
        self._blocks = []
        for i, dilation in enumerate(layers_list):
            self._blocks.append(
                ARBlock(
                    embed_dim + z_dim if i == 0 else external_size,
                    k_size, dilation,
                    internal_size=internal_size, external_size=external_size, do=do
                )
            )
        self.blocks = nn.Sequential(*self._blocks)
        self.pred_word = nn.Conv1d(external_size, vocab_size, 1)
        self.dropout = nn.Dropout(p=do)

    def forward(self, x, z):
        """

        :param x: sentence tensor embeddings from encoder [batch_size, seq_length, embedding_size]
        :param z: context tensor [batch_size, z_dim]
        :return:
        """
        # must permute to [batch_size, embedding_size, seq_length] for 1D conv
        x = x.permute(0, 2, 1).contiguous()
        # Add a single 0 example onto x
        x_dim_exp = x.view(x.size(0), x.size(1), x.size(2), 1).permute(0, 3, 1, 2)
        # need to swap the axis cuz this operation only works on images
        x_dim_exp_padded = F.pad(x_dim_exp, (1, 0, 0, 0))
        x_padded = x_dim_exp_padded[:, 0, :, :]
        # need to copy z (num_words + 1) times in the 2nd dimension then concat to x_padded
        z_exp = z.view(z.size(0), z.size(1), 1).expand(z.size(0), z.size(1), x.size(2) + 1)
        # concatenate z with x
        dec_input = torch.cat([x_padded, z_exp], 1)

        x_cur = self.blocks(dec_input)
        # crop off the last bit so we have probabilities
        # that match up with our input
        x_cur = x_cur[:, :, :x.size(2)]
        pred_words = self.pred_word(self.dropout(x_cur))
        return pred_words


if __name__ == "__main__":
    z = Variable(torch.zeros((10, 64)).normal_())
    x = Variable(torch.zeros((10, 7, 128)).normal_())

    dec = CNNDecoder(128, 64)
    dec.eval()
    out1 = dec(x, z)
    z2 = Variable(torch.zeros((10, 64)).normal_())
    out2 = dec(x, z2)
    same = (out1[:, :, 0]==out2[:, :, 0]).data.numpy().all()
    # for different z's and the same x, the first output should be different
    assert not same

    z = Variable(torch.zeros((10, 64)).normal_())
    x = Variable(torch.zeros((10, 7, 128)).normal_())

    out1 = dec(x, z)
    x2 = Variable(torch.zeros((10, 7, 128)).normal_())
    out2 = dec(x2, z)

    same = (out1[:, :, 0]==out2[:, :, 0]).data.numpy().all()
    # for different x and the same z, the first output should be the same
    assert same
