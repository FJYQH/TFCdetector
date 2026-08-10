import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_layer import AttentionLayer
from .embedding import TokenEmbedding, InputEmbedding

# ours
from .ours_memory_module import MemoryModule
# memae
# from .memae_memory_module import MemoryModule
# mnad
# from .mnad_memory_module import MemoryModule

# ### ours
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from .attn_layer import EncoderLayer_selfattn
import numpy as np
from torch.fft import rfft, irfft

_NEXT_FAST_LEN = {}

from utils.residual_loss import winsorization, gauss_filter


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.GroupNorm(4, out_channels, eps=1e-05, affine=True, device=None, dtype=None)  #
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class GRUBlock1d(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True, dropout=0.1, bidirectional=False):
        super(GRUBlock1d, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout,
                          bidirectional=bidirectional)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        #  B L C
        # # 初始化隐藏状态
        # h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        # 前向传播GRU
        out, _ = self.gru(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)
        return out


class Encoder1d(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super(Encoder1d, self).__init__()
        self.conv = ConvBlock1d(in_channels, out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) if pool else None

    def forward(self, x):
        x = self.conv(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class Decoder1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder1d, self).__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock1d(in_channels, out_channels)

    def forward(self, x1):
        x1 = self.upsample(x1)
        x = self.conv(x1)
        return x


class UNet1d_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet1d_2, self).__init__()
        self.inc = ConvBlock1d(in_channels, 16)
        self.enc1 = Encoder1d(16, 32, pool=True)
        self.enc2 = Encoder1d(32, 64, pool=True)
        self.center = ConvBlock1d(64, 64)

        self.dec2 = ConvBlock1d(64, 32)
        self.dec1 = ConvBlock1d(32, 16)
        self.outc = nn.Conv1d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.center(x3)

        x4_up = F.interpolate(x4, scale_factor=2, mode='linear')
        x4_up = x4_up[:, :, :x3.shape[-1]]
        x5 = self.dec2(x4_up)

        x5_up = F.interpolate(x5, scale_factor=2, mode='linear')
        x5_up = x5_up[:, :, :x2.shape[-1]]
        x6 = self.dec1(x5_up)
        x7 = self.outc(x6)
        return x7


class UNet1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet1d, self).__init__()
        self.inc = ConvBlock1d(in_channels, 16)
        self.enc1 = Encoder1d(16, 32, pool=True)
        self.enc2 = Encoder1d(32, 64, pool=True)
        self.center = ConvBlock1d(64, 64)
        self.dec2 = Decoder1d(64, 32)
        self.dec1 = Decoder1d(32, 16)
        self.outc = nn.Conv1d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.center(x3)

        x5 = self.dec2(x4)
        x6 = self.dec1(x5)
        x7 = self.outc(x6)
        return x7


class UNetGRU(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers, out_channels):
        super(UNetGRU, self).__init__()
        self.gru = GRUBlock1d(in_channels, in_channels, num_layers)

        self.enc_conv1 = nn.Conv1d(in_channels, hidden_size * 2, kernel_size=3, stride=2, padding=1)
        self.enc_gru1 = GRUBlock1d(hidden_size * 2, hidden_size * 2, num_layers)

        self.enc_conv2 = nn.Conv1d(hidden_size * 2, hidden_size * 4, kernel_size=3, stride=2, padding=1)
        self.enc_gru2 = GRUBlock1d(hidden_size * 4, hidden_size * 4, num_layers)

        self.dec_conv1 = nn.ConvTranspose1d(hidden_size * 4, hidden_size * 2, kernel_size=2, stride=2)
        self.dec_gru3 = GRUBlock1d(hidden_size * 2, hidden_size * 2, num_layers)

        self.dec_conv2 = nn.ConvTranspose1d(hidden_size * 2, out_channels, kernel_size=2, stride=2)
        self.dec_gru4 = GRUBlock1d(out_channels, out_channels, num_layers)

    def forward(self, x):
        x = self.gru(x)

        # Encoder path
        enc1 = F.relu(self.enc_conv1(x))
        enc1 = self.enc_gru1(enc1)
        enc2 = F.relu(self.enc_conv2(enc1))
        enc2 = self.enc_gru2(enc2)

        # Decoder path
        dec1 = self.dec_conv1(enc2)
        dec1 = self.dec_gru3(dec1)
        dec2 = F.relu(self.dec_conv2(dec1))
        dec2 = self.dec_gru4(dec2)

        return dec2


#########################################

class EncoderLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()

        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.attn_layer = attn
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        out = self.attn_layer(x)
        x = x + self.dropout(out)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)  # N x L x C(=d_model)


# Transformer Encoder
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        for attn_layer in self.attn_layers:
            x = attn_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x


class Decoder(nn.Module):
    def __init__(self, d_model, c_out, d_ff=None, activation='relu', dropout=0.1):
        super(Decoder, self).__init__()
        # self.decoder_layer = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=2,
        #                              batch_first=True, bidirectional=True)
        self.out_linear = nn.Linear(d_model, c_out)

    def forward(self, x):
        '''
        x : N x L x C(=d_model)   torch.Size([128, 96, 55])  permute(0, 2, 1)
        '''

        '''
        out : reconstructed output
        '''
        out = self.out_linear(x)
        return out  # N x L x c_out


class TransformerVar(nn.Module):
    # ##默认 dropout=0
    def __init__(self, win_size, enc_in, c_out, n_memory, shrink_thres=0, \
                 d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.1, activation='gelu', \
                 device=None, memory_init_embedding=None, memory_initial=False, phase_type=None, dataset_name=None):
        super(TransformerVar, self).__init__()

        self.memory_initial = memory_initial

        # Encoding
        self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout,
                                        device=device)  # N x L x C(=d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        win_size, d_model, n_heads, dropout=dropout
                    ), d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )
        self.mem_module = MemoryModule(n_memory=n_memory, fea_dim=d_model, shrink_thres=shrink_thres, device=device,
                                       memory_init_embedding=memory_init_embedding, phase_type=phase_type,
                                       dataset_name=dataset_name)

        # ours
        self.weak_decoder = Decoder(2 * d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)
        # baselines
        # self.weak_decoder = Decoder(d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)

        ############################################# 小波分解、单成分建模
        # self.dwt = DWT1DForward(wave='haar', J=3, mode='symmetric')
        self.dwt = DWT1DForward(wave='haar', J=3, mode='symmetric')

        self.idwt = DWT1DInverse(wave='haar')
        self.f_local_ad = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1)
        self.fuse_linear = nn.Linear(win_size * 2, win_size)

        self.unet1d_model = nn.ModuleList([UNet1d(enc_in, enc_in) for _ in range(4)])
        # self.gruunet = UNetGRU(enc_in, enc_in, 2, enc_in)

        ######################### 频率 全局、局部学习
        self.atten = nn.ModuleList(  # d_model", default=256   d_inner", default=512  n_head", default=8
            [
                EncoderLayer_selfattn(256, 512, 8, 64, 64, dropout=0.1,
                                      # self.hp.d_model,
                                      # self.hp.d_inner,
                                      # self.hp.n_head,
                                      # self.hp.d_inner // self.hp.n_head,
                                      # self.hp.d_inner // self.hp.n_head,
                                      # dropout=0.1,
                                      )
                for _ in range(1)
            ]
        )

        self.emb_local = nn.Sequential(  # kernel_size 24 --stride 8   d_model", default=256
            nn.Linear(2 + 24, 256),
            nn.Tanh(), )

        self.out_linear = nn.Sequential(  # condition_emb_dim 64   self.hp.d_model, self.hp.condition_emb_dim
            nn.Linear(256, 96),
            nn.Tanh(), )

        self.dropout = nn.Dropout(0.05)
        self.emb_global = nn.Sequential(  # window 48    #  condition_emb_dim 64
            nn.Linear(win_size, win_size),
            nn.Tanh(), )

        self.criterion = nn.L1Loss()

    def get_conditon(self, x):
        # 输入 ([128, 1, 96])
        # print('get_conditon_input:', x.shape)
        x_g = x
        f_global = torch.fft.rfft(x_g[:, :, :-1], dim=-1)  # 输入 ([128, 1, 48])
        f_global = torch.cat((f_global.real, f_global.imag), dim=-1)  # 输入 ([128, 1, 96])
        f_global = self.emb_global(f_global)  # ([128, 1, 96])

        x_g = x_g.view(x.shape[0], 1, 1, -1)  # 输入 ([128,1, 1, 96])
        x_l = x_g.clone()
        x_l[:, :, :, -1] = 0
        unfold = nn.Unfold(
            kernel_size=(1, 24),
            dilation=1,
            padding=0,
            stride=(1, 8), )

        unfold_x = unfold(x_l)  # ([128, 24, 10])

        unfold_x = unfold_x.transpose(1, 2)
        f_local = torch.fft.rfft(unfold_x, dim=-1)  # [128, 10,13]
        f_local = torch.cat((f_local.real, f_local.imag), dim=-1)  # [128, 10,26]

        f_local = self.emb_local(f_local)  # [128, 10, 256] 希望是 256

        for enc_layer in self.atten:
            f_local, enc_slf_attn = enc_layer(f_local)
            # f_local [128, 10, 256]

        f_local = self.out_linear(f_local)  # 调整尺寸 [128, 10, 96]
        f_local = self.f_local_ad(f_local)  # [128, 1, 96]
        ## f_local = f_local[:, -1, :].unsqueeze(1)  # [128, 1, 96]

        # output = torch.cat((f_global, f_local), -1)
        # print('f_local:',f_local.shape)  f_local: torch.Size([128, 1, 96])
        # print('f_global:', f_global.shape)  f_global: torch.Size([128, 8, 96])
        ##output = f_local + f_global                 # [128, 1, 96]

        output = self.fuse_linear(torch.cat((f_local, f_global), dim=-1))
        return output

    def forward(self, x):
        '''
        x (input time window) : N x L x enc_in
        '''
        # print("x:",x.shape)  ([128, 96, 55])

        enc_out = x
        tmp = enc_out.permute(0, 2, 1)  # B C L

        tmp_l, tmp_h = self.dwt(tmp)
        tmp_coefs = [tmp_l] + tmp_h

        new_coefs = []
        de_loss = 0
        for i, series in enumerate(tmp_coefs):
            # if i ==1:
            #     series_win = winsorization(series.cpu().numpy())
            #     # series = gauss_filter(series)
            #     series_tensor = torch.from_numpy(series_win).cuda()
            # if i==2:
            #     series_win = winsorization(series.cpu().numpy())
            #     #series = gauss_filter(series)
            #     series_tensor = torch.from_numpy(series_win).cuda()
            # if i!=1 and i!=2:
            #     series_tensor = series

            series_unet = self.unet1d_model[i](series)
            new_coefs.append(series_unet)
            L1_loss = torch.pow(series_unet - series, 2).mean()
            de_loss += L1_loss

        new_series_out_wave = self.idwt((new_coefs[0], new_coefs[1:]))

        # 序列分解后，重建后 处理--频域 学习
        new_series_out = torch.unbind(new_series_out_wave, dim=1)
        frequent_series_out = [self.get_conditon(series.unsqueeze(1)) for series in new_series_out]
        # print(frequent_series_out[0].shape)

        frequent_series_out = torch.stack(frequent_series_out, dim=1).squeeze()
        # print(frequent_series_out.shape)  #torch.Size([128, 8, 8, 96])

        # new_series_out = self.gruunet(frequent_series_out)
        new_series_out = frequent_series_out
        x = new_series_out.permute(0, 2, 1)

        ##########################################################################

        x = self.embedding(x)  # embeddin : N x L x C(=d_model)
        # print("x2:", x.shape)   ([128, 96, 512])

        queries = out = self.encoder(x)  # encoder out : N x L x C(=d_model)

        outputs = self.mem_module(out)
        out, attn, memory_item_embedding = outputs['output'], outputs['attn'], outputs['memory_init_embedding']
        mem = self.mem_module.mem

        if self.memory_initial:
            return {"out": out, "memory_item_embedding": None, "queries": queries, "mem": mem,
                    "new_series": new_series_out_wave.permute(0, 2, 1)}
        else:

            # print("out:",out.shape)  torch.Size([128, 96, 1024])
            out = self.weak_decoder(out)
            # print("out1:", out.shape)  torch.Size([128, 96, 55])

            '''
            out (reconstructed input time window) : N x L x enc_in
            enc_in == c_out
            '''
            return {"out": out, "memory_item_embedding": memory_item_embedding, "queries": queries, "mem": mem,
                    "attn": attn, "new_series": new_series_out_wave.permute(0, 2, 1), "de_loss": de_loss}
