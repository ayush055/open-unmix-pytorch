from tokenize import String
from typing import Optional, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter, Transformer
from .filtering import wiener
from .transforms import make_filterbanks, ComplexNorm
from .transformer import PositionalEncoding
from openunmix import transformer
import os
import torchaudio
from openunmix import utils

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

        self.pos_encoder = PositionalEncoding(hidden_size, dropout=0.5)

        # self.fc_decoder = Linear(self.nb_bins * nb_channels, hidden_size, bias=False)
        # self.bn_decoder = BatchNorm1d(hidden_size)
        self.transformer = Transformer(
            d_model=hidden_size,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dropout=0.5,
            activation='gelu',
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

    def forward(self, x: Tensor, y: Tensor, pass_y=False) -> Tensor:
        """
        Args:
            x: input spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`

        Returns:
            Tensor: filtered spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`
        """

        # change to correct shape:
        x = x.permute(3, 0, 1, 2)
        # frames, bathces, channels, bins
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        # so we learn a residual
        mix = x.detach().clone()

        # crop
        x = x[..., : self.nb_bins]
        # shift and scale input to mean=0 std=1 (across all bins)
        x = x + self.input_mean
        x = x * self.input_scale

        # convert (nb_frames, nb_samples, nb_channels*nb_bins)
        # to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range to [-1, 1]
        x = torch.tanh(x)
        SOS_TOKEN = torch.full((1, x.size(1), x.size(2)), 2, dtype=torch.float32)
        # EOS_TOKEN = torch.full((1, x.size(1), x.size(2)), 3, dtype=torch.float32)
        SOS_TOKEN = SOS_TOKEN.to(x.device)
        # EOS_TOKEN = EOS_TOKEN.to(x.device)
        x = torch.cat((SOS_TOKEN, x), dim=0)
        x = self.pos_encoder(x)

        if not pass_y:
            # same manipulation for y
            y = y.permute(3, 0, 1, 2)
            y_frames, y_samples, y_channels, y_bins = y.data.shape
            y = y[..., : self.nb_bins]
            y = y + self.input_mean
            y = y * self.input_scale
            # Frames x Samples x Frequency Domain
            y = self.fc1(y.reshape(-1, y_channels * self.nb_bins))
            y = self.bn1(y)
            y = y.reshape(y_frames, y_samples, self.hidden_size)
            y = torch.tanh(y)
            # SOS_TOKEN = torch.full((1, y.size(1), y.size(2)), 2, dtype=torch.float32)
            EOS_TOKEN = torch.full((1, y.size(1), y.size(2)), 3, dtype=torch.float32)
            # SOS_TOKEN = SOS_TOKEN.to(y.device)
            EOS_TOKEN = EOS_TOKEN.to(y.device)
            y = torch.cat((y, EOS_TOKEN), dim=0)
            y = self.pos_encoder(y)

        # TARGET MASK?? TODO
        sequence_length = y.size(0)
        tgt_mask = self.get_tgt_mask(sequence_length).to(self.device)

        transformer_out = self.transformer(x, y, tgt_mask=tgt_mask)
        # print("encoder input: ", x.size())
        # print("decoder target input: ", y.size())
        # print("transformer output: ", transformer_out.size())

        x = x[1:, :, :]
        # transformer_out = transformer_out[:-1, :, :]
        
        # print("x: ", x.size())
        # print("transformer_out: ", transformer_out.size())
        x = torch.cat([x, transformer_out], -1)
        # print("x cat: ", x.size())
        # print("x: ", `x.size())
        # x = nn.Dropout(0.5)(x)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix
        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        if pass_y:
            return x.permute(1, 2, 3, 0), y
        else:
            return x.permute(1, 2, 3, 0)

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask
    

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

        self.stft, self.istft = make_filterbanks(
            n_fft=n_fft,
            n_hop=n_hop,
            center=True,
            method=filterbank,
            sample_rate=sample_rate,
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

    def forward(self, audio: Tensor, track) -> Tensor:
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
        mix_stft = self.stft(audio)
        X = self.complexnorm(mix_stft)

        # initializing spectrograms variable
        spectrograms = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype, device=X.device)


        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            # create target
            SOS_TOKEN = torch.full((1, 1, 512), 2, dtype=torch.float32, device=X.device)
            y_input = SOS_TOKEN.clone().detach()
            
            # for i in range():
            print(X.size())
            for i in range(X.size(0)):

                pred, y = target_module(X.detach().clone(), y_input, pass_y=True)
                print("pred: ", pred.size())
                print("y: ", y.size())
                next_item = y.topk(1)[1].view(-1)[-1].item() # num with highest probability
                next_item = torch.tensor([[next_item]], device=X.device)

                # Concatenate previous input with predicted best word
                y_input = torch.cat((y, next_item), dim=1)

            target_spectrogram = target_module(X.detach().clone(), y_input, pass_y=True)
            spectrograms[..., j] = target_spectrogram

        # transposing it as
        # (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)
        spectrograms = spectrograms.permute(0, 3, 2, 1, 4)

        # rearranging it into:
        # (nb_samples, nb_frames, nb_bins, nb_channels, 2) to feed
        # into filtering methods
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
