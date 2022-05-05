import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def pad_layer(inp, layer, is_2d=False):
    if type(layer.kernel_size) == tuple:
        kernel_size = layer.kernel_size[0]
    else:
        kernel_size = layer.kernel_size
    if not is_2d:
        if kernel_size % 2 == 0:
            pad = (kernel_size//2, kernel_size//2 - 1)
        else:
            pad = (kernel_size//2, kernel_size//2)
    else:
        if kernel_size % 2 == 0:
            pad = (kernel_size//2, kernel_size//2 - 1, kernel_size//2, kernel_size//2 - 1)
        else:
            pad = (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(inp,
            pad=pad,
            mode='reflect')
    out = layer(inp)
    return out


def RNN(inp, layer):
    inp_permuted = inp.permute(2, 0, 1)
    state_mul = (int(layer.bidirectional) + 1) * layer.num_layers
    zero_state = Variable(torch.zeros(state_mul, inp.size(0), layer.hidden_size))
    zero_state = zero_state.cuda() if torch.cuda.is_available() else zero_state
    out_permuted, _ = layer(inp_permuted, zero_state)
    out_rnn = out_permuted.permute(1, 2, 0)
    return out_rnn

def linear(inp, layer):
    batch_size = inp.size(0)
    hidden_dim = inp.size(1)
    seq_len = inp.size(2)
    inp_permuted = inp.permute(0, 2, 1)
    inp_expand = inp_permuted.contiguous().view(batch_size*seq_len, hidden_dim)
    out_expand = layer(inp_expand)
    out_permuted = out_expand.view(batch_size, seq_len, out_expand.size(1))
    out = out_permuted.permute(0, 2, 1)
    return out


def pixel_shuffle_1d(inp, upscale_factor=2):
    batch_size, channels, in_width = inp.size()
    channels //= upscale_factor

    out_width = in_width * upscale_factor
    inp_view = inp.contiguous().view(batch_size, channels, upscale_factor, in_width)
    shuffle_out = inp_view.permute(0, 1, 3, 2).contiguous()
    shuffle_out = shuffle_out.view(batch_size, channels, out_width)
    return shuffle_out


class Encoder_f0(nn.Module):
    '''
        Map f0(B, T) to f0_emb(B, 64, T)
    '''
    def __init__(self, emb_lf0=True):
        super(Encoder_f0, self).__init__()
        self.emb_lf0 = emb_lf0
        self.conv_layers = nn.Sequential(nn.Conv1d(1, 256, kernel_size=5, stride=1, padding=2),
                                        nn.GroupNorm(256//16, 256),
                                        nn.ReLU(),
                                        nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
                                        nn.GroupNorm(256//16, 256),
                                        nn.ReLU(),
                                        nn.Conv1d(256, 256, kernel_size=1, stride=1, padding=0),
                                        nn.GroupNorm(256//16, 256),
                                        nn.ReLU())
        self.lstm = nn.LSTM(256, 64, 1, batch_first=True, bidirectional=True)
    def forward(self, lf0):
        lf0 = lf0.unsqueeze(1) # (B, 1, T)
        if self.emb_lf0:
            lf0 = self.conv_layers(lf0) # (B, 256, T)
            lf0 = lf0.permute(0, 2, 1) # (B, T, 256)
            lf0, _ = self.lstm(lf0) # (B, T, 64)
        else:
            lf0 = lf0.permute(0, 2, 1) #(B, T, 1)
        return lf0



class Encoder(nn.Module):
    '''
        Map mel(B, 80, T) to latent_z(B, 512, T)
    '''
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(80, 512, kernel_size=5, stride=1, padding=2)
        self.lin_layers = nn.Sequential(
            nn.LayerNorm(512),
            nn.ReLU(True),
            nn.Linear(512, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(True),
            nn.Linear(512, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(True),
            nn.Linear(512, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(True),
            nn.Linear(512, 64)
        )
        # self.lstm = nn.LSTM(64, 32, batch_first=True, bidirectional=True)
                                         
    def forward(self, mels):
        z = self.conv(mels) # (B, 512, T)
        z = self.lin_layers(z.permute(0, 2, 1)) # (B, T, 64)
        return z
        

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x    


class Decoder(nn.Module):
    def __init__(self, c_in=64, c_out=80, dim_pre=512, emb_size=1):
        super(Decoder, self).__init__()
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        # rnn layer
        self.lstm1 = nn.LSTM(c_in+emb_size, dim_pre, 1, batch_first=True)
        self.lstm2 = nn.LSTM(dim_pre, dim_pre*2, 1, batch_first=True)
        self.linear_projection = LinearNorm(dim_pre*2, c_out)
        # postnet
        self.postnet = Postnet()

    def forward(self, z, lf0_emb):
        # concat
        x = torch.cat([z, lf0_emb], dim=-1)
        x, _ = self.lstm1(x)
        x = x.permute(0, 2, 1)
        # conv layer
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.permute(0, 2, 1)
        # rnn layer
        out, _ = self.lstm2(x)
        out = self.linear_projection(out)
        out = out.permute(0, 2, 1)
        out_postnet = self.postnet(out)
        out_postnet = out + out_postnet
        return out, out_postnet
    
    def loss_function(self, mel, mel_pred, mel_pred_postnet, use_l1_loss=True):
        loss = F.mse_loss(mel, mel_pred) + F.mse_loss(mel, mel_pred_postnet)
        if use_l1_loss:
            return loss + F.l1_loss(mel, mel_pred) + F.l1_loss(mel, mel_pred_postnet)
        else:
            return loss


if __name__ == "__main__":
    device = torch.device('cuda:1')
    mels = torch.randn(256, 80, 128).to(device)
    lf0 = torch.randn(256, 128).to(device)

    f0_net = Encoder_f0(emb_lf0=False).to(device)
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    lf0_emb = f0_net(lf0)
    print(f"lf0_emb: {lf0_emb.shape}")

    z = encoder(mels)
    print(f"z: {z.shape}")

    _, outs = decoder(z, lf0_emb) 
    print(f"out: {outs.shape}")