from typing import Optional, Mapping

import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter, TransformerEncoderLayer
from .filtering import wiener
from .transforms import make_filterbanks, ComplexNorm

import math
from torch.autograd import Variable

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, kernel_size=4, in_channels=2, out_channels=64):
        super(Encoder, self).__init__()
        self.encoder = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size//2,
            padding=0,
            bias=False
        )
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.activation(self.encoder(x))

class Decoder(nn.Module):
    def __init__(self, kernel_size=4, in_channels=64, out_channels=2):
        super(Decoder, self).__init__()
        self.decoder = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size//2,
            padding=0,
            bias=False
        )
    
    def forward(self, x):
        return self.decoder(x)

class TransformerBlock(nn.Module):
    def __init__(self, feature_size, num_encoder_layers, num_heads):
        super(TransformerBlock, self).__init__()
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads

        self.pos_encoder_intra = PositionalEncoding(d_model=feature_size, dropout=0, max_len=20000)
        self.encoder_layers_intra = nn.ModuleList([])
        for _ in range(self.num_encoder_layers):
            self.encoder_layers_intra.append(
                TransformerEncoderLayer(d_model=feature_size, nhead=self.num_heads, dim_feedforward=feature_size*4, dropout=0)
            )

        self.pos_encoder_inter = PositionalEncoding(d_model=feature_size, dropout=0, max_len=20000)
        self.encoder_layers_inter = nn.ModuleList([])
        for _ in range(self.num_encoder_layers):
            self.encoder_layers_inter.append(
                TransformerEncoderLayer(d_model=feature_size, nhead=self.num_heads, dim_feedforward=feature_size*4, dropout=0)
            )
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [B, N, K, T]
        """

        B, N, K, T = x.shape

        # Temporal positional encoding injected based on time but encoder layers process chunks

        x_time = x.permute(0, 3, 2, 1).reshape(B*T, K, N) # Sequence length is num time steps
        x_intra = x_time + self.pos_encoder_intra(x_time)

        for i in range(self.num_encoder_layers):
            x_intra = self.encoder_layers_intra[i](x_intra.permute(1, 0, 2)).permute(1, 0, 2)

        x_out = x_intra + x_time
        x_out = x_out.reshape(B, T, K, N).permute(0, 3, 2, 1) # B, N, K, T

        x_chunk = x_out.permute(0, 2, 3, 1).reshape(B*K, T, N) # Sequence length is num chunks
        x_inter = x_chunk + self.pos_encoder_inter(x_chunk)

        for i in range(self.num_encoder_layers):
            x_inter = self.encoder_layers_inter[i](x_inter.permute(1, 0, 2)).permute(1, 0, 2)

        x_final = x_inter + x_chunk
        x_final = x_final.reshape(B, K, T, N).permute(0, 3, 1, 2) # B, N, K, T

        return x

class TransformerSeparator(nn.Module):
    def __init__(self, speakers, chunk_size, num_transformer_blocks, num_encoder_layers, num_heads, conv_filters):
        super(TransformerSeparator, self).__init__()
        self.speakers = speakers
        self.chunk_size = chunk_size
        self.num_transformer_blocks = num_transformer_blocks
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads
        self.conv_filters = conv_filters

        self.transformer_blocks = nn.ModuleList([])
        for _ in range(self.num_transformer_blocks):
            self.transformer_blocks.append(
                TransformerBlock(
                feature_size=self.conv_filters, 
                num_encoder_layers=self.num_encoder_layers,
                num_heads=self.num_heads,
                )
            )
        
        self.LayerNorm = nn.LayerNorm(conv_filters)
        self.Linear1 = nn.Linear(in_features=conv_filters, out_features=conv_filters, bias=None)

        self.PReLU = nn.PReLU()
        self.Linear2 = nn.Linear(in_features=conv_filters, out_features=conv_filters*2, bias=None)

        self.FeedForward1 = nn.Sequential(nn.Linear(conv_filters*2, conv_filters*2*2),
                                          nn.ReLU(),
                                          nn.Linear(conv_filters*2*2, conv_filters))
        self.FeedForward2 = nn.Sequential(nn.Linear(conv_filters, conv_filters*2*2),
                                          nn.ReLU(),
                                          nn.Linear(conv_filters*2*2, conv_filters))
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.LayerNorm(x.permute(0, 2, 1))
        x = self.Linear1(x).permute(0, 2, 1)

        out, gap = self.split_feature(x, self.chunk_size)

        for i in range(self.num_transformer_blocks):
            out = self.transformer_blocks[i](out)
        
        out = self.PReLU(out)
        # print("out shape: ", out.shape)
        out = self.Linear2(out.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        # print("out shape: ", out.shape)
        B, _, K, S = out.shape

        # OverlapAdd
        out = out.reshape(B, -1, self.speakers, K, S).permute(0, 2, 1, 3, 4)  # [B, N*C, K, S] -> [B, N, C, K, S]
        # print("out shape: ", out.shape)
        out = out.reshape(B * self.speakers, -1, K, S)
        # print("out shape: ", out.shape)
        out = self.merge_feature(out, gap)  # [B*C, N, K, S]  -> [B*C, N, I]

        # FFW + ReLU
        # print("separator end", out.shape)
        out = self.FeedForward1(out.permute(0, 2, 1))
        out = self.FeedForward2(out).permute(0, 2, 1)
        out = self.ReLU(out)

        return out

    # TAKEN FROM SEPFORMER CODE
    def pad_segment(self, input, segment_size):

        # 输入特征: (B, N, T)

        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size

        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())

        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def split_feature(self, input, segment_size):

        # 将特征分割成段大小的块
        # 输入特征: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        segments2 = input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2, 3)

        return segments.contiguous(), rest

    def merge_feature(self, input, rest):

        # 将分段的特征合并成完整的话语
        # 输入特征: (B, N, L, K)

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)  # B, N, K, L

        input1 = input[:, :, :, :segment_size].contiguous().view(batch_size, dim, -1)[:, :, segment_stride:]
        input2 = input[:, :, :, segment_size:].contiguous().view(batch_size, dim, -1)[:, :, :-segment_stride]

        output = input1 + input2

        if rest > 0:
            output = output[:, :, :-rest]

        return output.contiguous()  # B, N, T

class TransformerWaveform(nn.Module):
    def __init__(self, speakers, input_channels, conv_kernel_size, conv_filters, chunk_size, num_transformer_blocks, num_encoder_layers, num_heads):
        super(TransformerWaveform, self).__init__()
        self.speakers = speakers
        self.in_channels = input_channels
        self.chunk_size = chunk_size
        self.num_transformer_blocks = num_transformer_blocks
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size

        self.encoder = Encoder(kernel_size=conv_kernel_size, in_channels=input_channels, out_channels=conv_filters)
        self.transformer = TransformerSeparator(speakers=speakers, chunk_size=chunk_size, num_transformer_blocks=num_transformer_blocks, num_encoder_layers=num_encoder_layers, num_heads=num_heads, conv_filters=conv_filters)
        self.decoder = Decoder(kernel_size=conv_kernel_size, in_channels=conv_filters, out_channels=input_channels)

    def forward(self, x):
        x, rest = self.pad_signal(x)

        enc_out = self.encoder(x)
        # print("enc_out: ", enc_out.shape)
        masks = self.transformer(enc_out)
        # print("masks: ", masks.shape)
        _, N, I = masks.shape

        masks = masks.view(self.speakers, -1, N, I)  # [C, B, N, I]，torch.Size([2, 1, 64, 16002])
        # print("masks reshaped: ", masks.shape)
        # print("MASKS", masks.shape)
        # print("Encoding", enc_out.shape)

        # Masking
        out = [masks[i] * enc_out for i in range(self.speakers)]  # C * ([B, N, I]) * [B, N, I]
        # print("out0", out[0].shape)
        # print("out1", out[1].shape)

        # Decoding
        audio = [self.decoder(out[i]) for i in range(self.speakers)]  # C * [B, 1, T]

        # print("audio0", audio[0].shape)
        # print("audio1", audio[1].shape)

        # for i in range(self.in_channels):
        audio[0] = audio[0][:, :, self.conv_kernel_size // 2:-(rest + self.conv_kernel_size // 2)].contiguous()  # B, 1, T
        # audio[1] = audio[1][:, :, self.conv_kernel_size // 2:-(rest + self.conv_kernel_size // 2)].contiguous()  # B, 1, T
        # audio = torch.cat(audio, dim=1)  # [B, C, T]

        return audio[0]

    def pad_signal(self, input):

        # 输入波形: (B, T) or (B, 1, T)
        # 调整和填充

        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)

        batch_size = input.size(0)  # 每一个批次的大小
        nsample = input.size(2)  # 单个数据的长度
        rest = self.conv_kernel_size - (self.conv_kernel_size // 2 + nsample % self.conv_kernel_size) % self.conv_kernel_size

        if rest > 0:
            pad = Variable(torch.zeros(batch_size, self.in_channels, rest)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        pad_aux = Variable(torch.zeros(batch_size, self.in_channels, self.conv_kernel_size // 2)).type(input.type())

        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

class OpenUnmix(nn.Module):
    """OpenUnmix Core spectrogram based separation module.

    Args:
        nb_bins (int): Number of input time-frequency bins (Default: `4096`).
        nb_channels (int): Number of input audio channels (Default: `2`).
        hidden_size (int): Size for bottleneck layers (Default: `512`).
        nb_layers (int): Number of Bi-LSTM layers (Default: `3`).
        unidirectional (bool): Use causal model useful for realtime purpose.
            (Default `False`)
        input_mean (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to zeros(nb_bins)
        input_scale (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to ones(nb_bins)
        max_bin (int or None): Internal frequency bin threshold to
            reduce high frequency content. Defaults to `None` which results
            in `nb_bins`
    """

    def __init__(
        self,
        nb_bins: int = 4096,
        nb_channels: int = 2,
        hidden_size: int = 512,
        nb_layers: int = 3,
        unidirectional: bool = False,
        input_mean: Optional[np.ndarray] = None,
        input_scale: Optional[np.ndarray] = None,
        max_bin: Optional[int] = None,
    ):
        super(OpenUnmix, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.nb_output_bins = nb_bins
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.hidden_size = hidden_size

        self.fc1 = Linear(self.nb_bins * nb_channels, hidden_size, bias=False)

        self.bn1 = BatchNorm1d(hidden_size)

        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4 if nb_layers > 1 else 0,
        )

        fc2_hiddensize = hidden_size * 2
        self.fc2 = Linear(in_features=fc2_hiddensize, out_features=hidden_size, bias=False)

        self.bn2 = BatchNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins * nb_channels,
            bias=False,
        )

        self.bn3 = BatchNorm1d(self.nb_output_bins * nb_channels)

        # Time Domain Layers

        self.transformer = TransformerWaveform(
            speakers=1,
            input_channels=2,
            conv_kernel_size=4,
            conv_filters=128,
            num_heads=4,
            chunk_size=250,
            num_transformer_blocks=2,
            num_encoder_layers=2,
        )

        self.filter_bins = Linear(in_features=self.nb_output_bins*nb_channels*2, out_features=self.nb_output_bins*nb_channels, bias=False)

        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean[: self.nb_bins]).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale[: self.nb_bins]).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = Parameter(torch.ones(self.nb_output_bins).float())

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: Tensor, x_time: Tensor) -> Tensor:
        """
        Args:
            x: input spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`

        Returns:
            Tensor: filtered spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`
        """

        # Time Domain code

        # print("x_time shape: ", x_time.shape)
        x_time = self.transformer(x_time)
        # print("x_time shape:", x_time.shape)
        resample = torchaudio.transforms.Resample(16000, 44100).to(self.device)
        x_time = resample(x_time)
        print("x_time shape:", x_time.shape)
        stft, _ = make_filterbanks(
        n_fft=4096, n_hop=1024, sample_rate=44100
        )
        encoder = torch.nn.Sequential(stft, ComplexNorm(mono=False)).to(self.device)
        x_time = encoder(x_time)
        print("x_time shape:", x_time.shape)

        # permute so that batch is last for lstm
        x = x.permute(3, 0, 1, 2)
        # get current spectrogram shape
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        mix = x.detach().clone()

        # crop
        x = x[..., : self.nb_bins]
        # shift and scale input to mean=0 std=1 (across all bins)
        x = x + self.input_mean
        x = x * self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)

        # lstm skip connection
        x = torch.cat([x, lstm_out[0]], -1)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # x_time = x_time.reshape()

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix

        # print("x time shape", x_time.shape)
        x_time = x_time.reshape(-1, x_time.shape[-1]).permute(1, 0)
        # print("x time shape", x_time.shape)
        x_time = x_time.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)
        # print("x time shape", x_time.shape)

        # x = (x + x_time) / 2
        x = torch.cat([x, x_time], -1)
        # x = nn.Dropout(0.25)(x)
        x = x.reshape(-1, self.nb_output_bins*nb_channels*2)
        x = self.filter_bins(x)
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)

        return x.permute(1, 2, 3, 0)


class Separator(nn.Module):
    """
    Separator class to encapsulate all the stereo filtering
    as a torch Module, to enable end-to-end learning.

    Args:
        targets (dict of str: nn.Module): dictionary of target models
            the spectrogram models to be used by the Separator.
        niter (int): Number of EM steps for refining initial estimates in a
            post-processing stage. Zeroed if only one target is estimated.
            defaults to `1`.
        residual (bool): adds an additional residual target, obtained by
            subtracting the other estimated targets from the mixture,
            before any potential EM post-processing.
            Defaults to `False`.
        wiener_win_len (int or None): The size of the excerpts
            (number of frames) on which to apply filtering
            independently. This means assuming time varying stereo models and
            localization of sources.
            None means not batching but using the whole signal. It comes at the
            price of a much larger memory usage.
        filterbank (str): filterbank implementation method.
            Supported are `['torch', 'asteroid']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.
    """

    def __init__(
        self,
        target_models: Mapping[str, nn.Module],
        niter: int = 0,
        softmask: bool = False,
        residual: bool = False,
        sample_rate: float = 44100.0,
        n_fft: int = 4096,
        n_hop: int = 1024,
        nb_channels: int = 2,
        wiener_win_len: Optional[int] = 300,
        filterbank: str = "torch",
    ):
        super(Separator, self).__init__()

        # saving parameters
        self.niter = niter
        self.residual = residual
        self.softmask = softmask
        self.wiener_win_len = wiener_win_len

        self.stft2, self.istft2 = make_filterbanks(
            n_fft=n_fft,
            n_hop=n_hop,
            center=True,
            method=filterbank,
            sample_rate=sample_rate,
        )

        nfft = 4096
        nhop = 1024

        self.stft, self.istft = make_filterbanks(
            n_fft=nfft, n_hop=nhop,
        )
        self.complexnorm = ComplexNorm(mono=nb_channels == 1)

        # registering the targets models
        self.target_models = nn.ModuleDict(target_models)
        # adding till https://github.com/pytorch/pytorch/issues/38963
        self.nb_targets = len(self.target_models)
        # get the sample_rate as the sample_rate of the first model
        # (tacitly assume it's the same for all targets)
        self.register_buffer("sample_rate", torch.as_tensor(sample_rate))

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def encoder_y(self, nfft, batch_size, nb_channels, seq_dur, nhop, encoder, arr_len, y, only_stft=False):
        bins = nfft // 2 + 1
        batch = batch_size
        channel = nb_channels
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        width = int(44100 * seq_dur)
        hop_length = width//2
        num_timesteps = y.size(-1)
        frame = 0
        
        if only_stft:
            arr = torch.zeros(size=(batch, channel, bins, arr_len, 2)).to(device)
        else:
            arr = torch.zeros(size=(batch, channel, bins, arr_len)).to(device)

        for i in range(0, num_timesteps, hop_length):
            y_tmp = y[..., i:(i + width)]

            if i + width > num_timesteps:
                # print("Time steps left", y_tmp.size(-1))
                num_frames_to_keep = int((y_tmp.size(-1) - (nfft - 1) - 1) / nhop) + 1
                padding = (0, i + width - num_timesteps)
                y_tmp = F.pad(y_tmp, padding, "constant", 0)
                y_tmp = encoder(y_tmp)
                if only_stft:
                    arr[..., frame:frame+num_frames_to_keep, :] += y_tmp[..., :num_frames_to_keep, :]    
                else:
                    arr[..., frame:frame+num_frames_to_keep] += y_tmp[..., :num_frames_to_keep]
                break
            
            y_tmp = encoder(y_tmp)
            if only_stft:
                print(arr.shape, y_tmp.shape)
                print(arr[..., frame:(frame + y_tmp.shape[-2]), :].shape)
                arr[..., frame:(frame + y_tmp.shape[-2]), :] += y_tmp
                frame += y_tmp.shape[-2] // 2
            else:
                arr[..., frame:(frame + y_tmp.shape[-1])] += y_tmp
                frame += y_tmp.shape[-1] // 2

        if only_stft:
            arr[..., :y_tmp.shape[-2] // 2, :] *= 2
            arr[..., frame + y_tmp.shape[-2] // 2:, :] *= 2
        else:
            arr[..., :y_tmp.shape[-1] // 2] *= 2
            arr[..., frame + y_tmp.shape[-1] // 2:] *= 2

        arr /= 2

        # print("original arr shape", arr.shape)
        if only_stft:
            arr = arr[..., :frame + num_frames_to_keep, :]
        else:
            arr = arr[..., :frame + num_frames_to_keep]

        # print("Final Y shape", arr.shape)

        return arr

    def decode_y(self, nfft, batch_size, nb_channels, seq_dur, nhop, encoder, num_timesteps, y):
        bins = nfft // 2 + 1
        batch = batch_size
        channel = nb_channels
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        width = int(44100 * seq_dur)
        hop_length = width//2
        frame = 0
        num_frames = y.shape[-1]

        num_windows = (num_timesteps // hop_length) + 1
        window_length = int((width - (nfft - 1) - 1) / nhop) + 1
        window_hop = window_length // 2

        arr = torch.zeros(size=(batch, channel, num_timesteps)).to(device)

        print(num_frames, window_hop)

        for i in range(0, num_frames, window_hop):
            y_tmp = y[..., i:(i + window_length)]



        self.istft(targets_stft, length=audio.shape[2])

    def forward(self, audio: Tensor) -> Tensor:
        """Performing the separation on audio input

        Args:
            audio (Tensor): [shape=(nb_samples, nb_channels, nb_timesteps)]
                mixture audio waveform

        Returns:
            Tensor: stacked tensor of separated waveforms
                shape `(nb_samples, nb_targets, nb_channels, nb_timesteps)`
        """

        nb_sources = self.nb_targets
        nb_samples = audio.shape[0]

        # getting the STFT of mix:
        # (nb_samples, nb_channels, nb_bins, nb_frames, 2)

        x = audio.clone()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq_dur = 4

        width = int(44100 * seq_dur)
        hop_length = width//2
        resample = torchaudio.transforms.Resample(44100, 16000).to(device)

        nfft = 4096
        nhop = 1024
        batch_size = 1
        nb_channels = 2
        
        encoder = torch.nn.Sequential(self.stft, ComplexNorm(mono=nb_channels == 1)).to(device)

        num_timesteps = x.size(-1)
        frame = 0
        num_windows = (num_timesteps // hop_length) + 1
        window_length = int((width - (nfft - 1) - 1) / nhop) + 1
        arr_len = (num_windows * window_length) // 2
        bins = nfft // 2 + 1
        batch = batch_size
        channel = nb_channels

        # initializing spectrograms variable
        # spectrograms = torch.zeros(size=(batch, channel, bins, arr_len) + (nb_sources,), dtype=audio.dtype, device=device)
        # print("SPECTOGRAMS SHAPE", spectrograms.shape)

        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            # apply current model to get the source spectrogram

            arr = torch.zeros(size=(batch, channel, bins, arr_len)).to(device)
            
            for i in range(0, num_timesteps, hop_length):
                X_tmp = x[..., i:(i + width)]
                x_time_temp = X_tmp.clone()

                if i + width > num_timesteps:
                    # print("Time steps left", X_tmp.size(-1))
                    num_frames_to_keep = int((X_tmp.size(-1) - (nfft - 1) - 1) / nhop) + 1
                    padding = (0, i + width - num_timesteps)
                    X_tmp, x_time_temp = F.pad(X_tmp, padding, "constant", 0), F.pad(x_time_temp, padding, "constant", 0)
                    X_tmp = encoder(X_tmp)
                    x_time_temp = resample(x_time_temp)

                    Y_hat = target_module(X_tmp, x_time_temp)
                    # print("Y_hat shape", Y_hat.shape)
                    print("i", i, "width", width, "num timesteps", num_timesteps, "frame", frame, "hop_length", hop_length)
                    # print("Keeping only {} frames".format(num_frames_to_keep))
                    arr[..., frame:frame+num_frames_to_keep] += Y_hat[..., :num_frames_to_keep]
                    # print("Final iteration start frame {}, end frame {}".format(frame, frame + num_frames_to_keep))
                    break
                
                X_tmp = encoder(X_tmp)
                x_time_temp = resample(x_time_temp)
                print(X_tmp.shape, x_time_temp.shape)
                Y_hat = target_module(X_tmp, x_time_temp)
                # print("Y_hat shape", Y_hat.shape)

                arr[..., frame:(frame + Y_hat.shape[-1])] += Y_hat
                frame += Y_hat.shape[-1] // 2
                # print("Frame start", frame)

            # print("arr shape", arr.shape)

            arr[..., :Y_hat.shape[-1] // 2] *= 2
            arr[..., frame + Y_hat.shape[-1] // 2:] *= 2
            # print("doubling frames from 0 to {} and from {} to end".format(frame, frame + Y_hat.shape[-1] // 2))
            
            arr /= 2

            # print("original arr shape", arr.shape)
            arr = arr[..., :frame + num_frames_to_keep]

            spectrograms = torch.zeros(size=(batch, channel, bins, arr.shape[-1]) + (nb_sources,), dtype=audio.dtype, device=device)
            print("arr shape", arr.shape)
            print("spectogram shape", spectrograms.shape)
            spectrograms[..., j] = arr
        
        # transposing it as
        # (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)
        spectrograms = spectrograms.permute(0, 3, 2, 1, 4)

        # rearranging it into:
        # (nb_samples, nb_frames, nb_bins, nb_channels, 2) to feed
        # into filtering methods

        mix_stft = self.encoder_y(nfft, batch_size, nb_channels, seq_dur, nhop, self.stft, arr.shape[-1], audio, only_stft=True)
        print(mix_stft.shape)
        
        mix_stft = mix_stft.permute(0, 3, 2, 1, 4)

        # create an additional target if we need to build a residual
        if self.residual:
            # we add an additional target
            nb_sources += 1

        if nb_sources == 1 and self.niter > 0:
            raise Exception(
                "Cannot use EM if only one target is estimated."
                "Provide two targets or create an additional "
                "one with `--residual`"
            )

        nb_frames = spectrograms.shape[1]
        targets_stft = torch.zeros(
            mix_stft.shape + (nb_sources,), dtype=audio.dtype, device=mix_stft.device
        )
        for sample in range(nb_samples):
            pos = 0
            if self.wiener_win_len:
                wiener_win_len = self.wiener_win_len
            else:
                wiener_win_len = nb_frames
            while pos < nb_frames:
                cur_frame = torch.arange(pos, min(nb_frames, pos + wiener_win_len))
                pos = int(cur_frame[-1]) + 1

                targets_stft[sample, cur_frame] = wiener(
                    spectrograms[sample, cur_frame],
                    mix_stft[sample, cur_frame],
                    self.niter,
                    softmask=self.softmask,
                    residual=self.residual,
                )

        # getting to (nb_samples, nb_targets, channel, fft_size, n_frames, 2)
        targets_stft = targets_stft.permute(0, 5, 3, 2, 1, 4).contiguous()

        # inverse STFT
        estimates = self.decode_y(nfft, batch_size, nb_channels, seq_dur, nhop, self.istft, audio.shape[-1], targets_stft)
        estimates = self.istft(targets_stft, length=audio.shape[2])

        return estimates

    def to_dict(self, estimates: Tensor, aggregate_dict: Optional[dict] = None) -> dict:
        """Convert estimates as stacked tensor to dictionary

        Args:
            estimates (Tensor): separated targets of shape
                (nb_samples, nb_targets, nb_channels, nb_timesteps)
            aggregate_dict (dict or None)

        Returns:
            (dict of str: Tensor):
        """
        estimates_dict = {}
        for k, target in enumerate(self.target_models):
            estimates_dict[target] = estimates[:, k, ...]

        # in the case of residual, we added another source
        if self.residual:
            print(self.residual)
            # estimates_dict["residual"] = estimates[:, -1, ...]
            estimates_dict["accompaniment"] = estimates[:, -1, ...]

        if aggregate_dict is not None:
            new_estimates = {}
            for key in aggregate_dict:
                new_estimates[key] = torch.tensor(0.0)
                for target in aggregate_dict[key]:
                    new_estimates[key] = new_estimates[key] + estimates_dict[target]
            estimates_dict = new_estimates
        return estimates_dict
