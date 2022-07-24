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

        # self.fc1 = Linear(self.nb_bins * nb_channels, hidden_size, bias=False)

        # self.bn1 = BatchNorm1d(hidden_size)

        self.pos_encoder = PositionalEncoding(self.nb_bins * nb_channels, dropout=0.5, max_len=257)

        # if unidirectional:
        #     lstm_hidden_size = hidden_size
        # else:
        #     lstm_hidden_size = hidden_size // 2

        # self.lstm = LSTM(
        #     input_size=hidden_size,
        #     hidden_size=lstm_hidden_size,
        #     num_layers=nb_layers,
        #     bidirectional=not unidirectional,
        #     batch_first=False,
        #     dropout=0.4 if nb_layers > 1 else 0,
        # )

        # custom_decoder_layer = CustomTransformerDecoder(nb_bins, nb_channels, hidden_size, d_model=hidden_size, nhead=8)
        # decoder_norm = LayerNorm(hidden_size, eps=1e-5)
        # self.decoder = TransformerDecoder(decoder_layer=custom_decoder_layer, num_layers=6, norm=decoder_norm)
        # self.pos_encoder_1 = PositionalEncoding(self.nb_bins * nb_channels, dropout=0.25)
        # self.pos_encoder_2 = PositionalEncoding(hidden_size, dropout=0.5)
        # self.fc_decoder = Linear(self.nb_bins * nb_channels, hidden_size, bias=False)
        # self.bn_decoder = BatchNorm1d(hidden_size)

        self.y_dropout = nn.Dropout(0.5)

        self.transformer = Transformer(
            d_model=self.nb_bins * nb_channels,
            nhead=3,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dropout=0.1,
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

    def forward(self, x: Tensor, y: Tensor, train=True) -> Tensor:
        """
        Args:
            x: input spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`
        Returns:
            Tensor: filtered spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`
        """

        # permute so that batch is last for lstm
        x = x.permute(3, 0, 1, 2)
        # get current spectrogram shape

        # samples (batch size), frames (duration of sequence, each frame is a time-bin of the STFT (~20ms for 5s duration, 255 frames), channels * bins is our frequency data

        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape
        # print("X shape:", nb_frames, nb_samples, nb_channels, nb_bins)
        # print("Y shape:", y_frames, y_samples, y_channels, y_bins)

        mix = x.detach().clone()

        # crop
        x = x[..., : self.nb_bins]
        # shift and scale input to mean=0 std=1 (across all bins)
        x = x + self.input_mean
        x = x * self.input_scale

        y = y.permute(3, 0, 1, 2)
        y_frames, y_samples, y_channels, y_bins = y.data.shape
        y = y[..., : self.nb_bins]
        y = y + self.input_mean
        y = y * self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)

        # print("X shape before first fc layer:", x.size())
        # x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        # normalize every instance in a batch
        # x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, nb_channels * self.nb_bins)
        # squash range to [-1, 1]
        # x = torch.tanh(x)

        # Samples x Frames x Hidden Size
        # x = np.swapaxes(x, 0, 1)

        SOS_TOKEN = torch.full((1, x.size(1), x.size(2)), -1, dtype=torch.float32)
        EOS_TOKEN = torch.full((1, x.size(1), x.size(2)), 0, dtype=torch.float32)
        SOS_TOKEN = SOS_TOKEN.to(x.device)
        EOS_TOKEN = EOS_TOKEN.to(x.device)

        # print("SOS shape:", SOS_TOKEN.size())
        # print("EOS shape:", EOS_TOKEN.size())

        x = torch.cat((SOS_TOKEN, x, EOS_TOKEN), dim=0)

        # x = nn.Dropout(0.5)(x)

        # Frames x Samples x Hidden Size
        # x = np.swapaxes(x, 0, 1)

        # x = self.pos_encoder(x)

        # Samples * Frames x Hidden Size
        # x = x.reshape(nb_samples * nb_frames, self.hidden_size)


        # print("X shape after first fc layer:", x.size())
        x = self.pos_encoder(x)

        # apply 3-layers of stacked LSTM
        # lstm_out = self.lstm(x)

        # print("Y shape before fc layer:", y.size())

        # Frames x Samples x Frequency Domain
        # if not transformer_only:
            # if not predict:
            #     if torch.rand(1) > 0.25:
            #         noise = torch.randn(y.size()).to(self.device)
            #         print("Adding noise to y tensor")
            #         # add noise to y
            #         y += noise
        # y = self.y_dropout(y)
        y = y.reshape(y_frames, y_samples, nb_channels * self.nb_bins)
            # print("X shape:", x.size())
            # print("Y shape", y.size())
        # y = torch.tanh(y)

            # y = y.reshape(y_frames, y_samples, 512)
            # y = y.reshape(y_frames * y_samples, 512)

            # # Samples x Frames x Hidden Size
            # y = np.swapaxes(y, 0, 1)

        if train:
            SOS_TOKEN = torch.full((1, y.size(1), y.size(2)), -1, dtype=torch.float32)
            EOS_TOKEN = torch.full((1, y.size(1), y.size(2)), 0, dtype=torch.float32)
            SOS_TOKEN = SOS_TOKEN.to(y.device)
            EOS_TOKEN = EOS_TOKEN.to(y.device)

            y = torch.cat((SOS_TOKEN, y, EOS_TOKEN), dim=0)

        # Frames x Samples x Hidden Size
        # y = np.swapaxes(y, 0, 1)
        y = self.pos_encoder(y)

        # y = y.reshape(-1, y_channels * self.nb_bins)

        # y_input = y[:, :-1]

        if train:
            y_input = y[:-1, ...]
            sequence_length = y_input.size(0)
            tgt_mask = self.get_tgt_mask(sequence_length).to(self.device)
            transformer_out = self.transformer(x, y_input, tgt_mask=tgt_mask)[:-1, ...]
            print(x.shape, y_input.shape, transformer_out.shape)
        else:
            sequence_length = y.size(0)
            tgt_mask = self.get_tgt_mask(sequence_length).to(self.device)
            transformer_out = self.transformer(x, y, tgt_mask=tgt_mask)

        # print(sequence_length, y_input.size(), tgt_mask.size())
        # print("Y shifted shape", y_input.size())

        # y_size = (y_frames, y_samples, y_input.size(-1))
        # x_size = (nb_frames, nb_samples, x.size(-1))
        # y = torch.tanh(y)
        
        # print(f"Y shape before encoder: {y.size()} \n {y}\n")
        # y = self.pos_encoder_1(y)
        # print(f"Y shape after encoder: {y.size()} \n {y}")
        # y = y.reshape(-1, y_channels * self.nb_bins)
        # y = self.fc_decoder(y)
        # y = self.bn_decoder(y)
        # y = y.reshape(y_frames, y_samples, self.hidden_size)
        # y = torch.tanh(y)
        # y = self.pos_encoder_2(y)

        # print("Target:", tgt.size(), tgt)

        # print("Y shape transformed:", y.shape)
        # tgt = torch.zeros(nb_frames, nb_samples, self.hidden_size).cuda()
        # print("Transformer out shape", transformer_out.size())
        # print(transformer_out.size())

        # lstm skip connection
        # x = torch.cat([x, lstm_out[0]], -1)
        # print("Transformer out:", transformer_only.size())
        # print("X shape:", x.size())

        # print("Transformer only", transformer_only)
        
        # x = torch.cat([x, transformer_out], -1)
        # x = nn.Dropout(0.5)(x)

        # first dense stage + batch norm
        # x = self.fc2(x.reshape(-1, x.shape[-1]))
        # x = self.bn2(x)

        # x = F.relu(x)
        # x = nn.Dropout(0.5)(x)

        # second dense stage + layer norm
        # x = self.fc3(x)
        # x = self.bn3(x)

        # reshape back to original dim
        x = transformer_out.reshape(-1, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        if not train:
            return x.permute(1, 2, 3, 0)

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix
        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        return x.permute(1, 2, 3, 0)

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)


    # def predict(self, x: Tensor, y: Tensor, tgt_mask: Tensor):
    #     """
    #     Args:
    #         x: input spectrogram of shape
    #             `(nb_samples, nb_channels, nb_bins, nb_frames)`

    #     Returns:
    #         Tensor: filtered spectrogram of shape
    #             `(nb_samples, nb_channels, nb_bins, nb_frames)`
    #     """

    #     # permute so that batch is last for lstm
    #     x = x.permute(3, 0, 1, 2)
    #     # get current spectrogram shape

    #     # samples (batch size), frames (duration of sequence, each frame is a time-bin of the STFT (~20ms for 5s duration, 255 frames), channels * bins is our frequency data

    #     nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape
    #     # print("X shape:", nb_frames, nb_samples, nb_channels, nb_bins)
    #     # print("Y shape:", y_frames, y_samples, y_channels, y_bins)

    #     mix = x.detach().clone()

    #     # crop
    #     x = x[..., : self.nb_bins]
    #     # shift and scale input to mean=0 std=1 (across all bins)
    #     x = x + self.input_mean
    #     x = x * self.input_scale

    #     # to (nb_frames*nb_samples, nb_channels*nb_bins)
    #     # and encode to (nb_frames*nb_samples, hidden_size)

    #     # print("X shape before first fc layer:", x.size())
    #     x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
    #     # normalize every instance in a batch
    #     x = self.bn1(x)
    #     x = x.reshape(nb_frames, nb_samples, self.hidden_size)
    #     # squash range to [-1, 1]
    #     x = torch.tanh(x)

    #     # Samples x Frames x Hidden Size
    #     # x = np.swapaxes(x, 0, 1)

    #     SOS_TOKEN = torch.full((1, x.size(1), x.size(2)), 2, dtype=torch.float32)
    #     EOS_TOKEN = torch.full((1, x.size(1), x.size(2)), 3, dtype=torch.float32)
    #     SOS_TOKEN = SOS_TOKEN.to(x.device)
    #     EOS_TOKEN = EOS_TOKEN.to(x.device)

    #     # print("SOS shape:", SOS_TOKEN.size())
    #     # print("EOS shape:", EOS_TOKEN.size())

    #     x = torch.cat((SOS_TOKEN, x, EOS_TOKEN), dim=0)

    #     # Frames x Samples x Hidden Size
    #     # x = np.swapaxes(x, 0, 1)

    #     x = self.pos_encoder(x)

    #     # print("X shape after encoder:", x.size())
    #     # print("Y shape:", y.size())
    #     # print("Tgt mask shape:", tgt_mask.size())
    #     transformer_out = self.transformer(x, y, tgt_mask=tgt_mask)
    #     # print(transformer_out.size())

    #     # lstm skip connection
    #     # x = torch.cat([x, lstm_out[0]], -1)
    #     # print("Transformer out:", transformer_out.size())
    #     # print("X shape:", x.size())

    #     x = x[1:-1, :, :]
    #     # print("Transformer out:", transformer_out.size())
    #     # print("X shape:", x.size())
    #     # transformer_out = transformer_out[1:, :, :]
    #     x = torch.cat([x, transformer_out], -1)

    #     # first dense stage + batch norm
    #     x = self.fc2(x.reshape(-1, x.shape[-1]))
    #     x = self.bn2(x)

    #     x = F.relu(x)

    #     # second dense stage + layer norm
    #     x = self.fc3(x)
    #     x = self.bn3(x)

    #     # reshape back to original dim
    #     x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

    #     # apply output scaling
    #     x *= self.output_scale
    #     x += self.output_mean

    #     # since our output is non-negative, we can apply RELU
    #     x = F.relu(x) * mix
    #     # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
    #     return x.permute(1, 2, 3, 0)

    def feed_transformer(self, x: Tensor, y: Tensor, tgt_mask: Tensor):
        """
        Args:
            x: input spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`

        Returns:
            Tensor: filtered spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`
        """

        # permute so that batch is last for lstm
        x = x.permute(3, 0, 1, 2)
        # get current spectrogram shape

        # samples (batch size), frames (duration of sequence, each frame is a time-bin of the STFT (~20ms for 5s duration, 255 frames), channels * bins is our frequency data

        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape
        # print("X shape:", nb_frames, nb_samples, nb_channels, nb_bins)
        # print("Y shape:", y_frames, y_samples, y_channels, y_bins)

        mix = x.detach().clone()

        # crop
        x = x[..., : self.nb_bins]
        # shift and scale input to mean=0 std=1 (across all bins)
        x = x + self.input_mean
        x = x * self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)

        # print("X shape before first fc layer:", x.size())
        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range to [-1, 1]
        x = torch.tanh(x)

        # Samples x Frames x Hidden Size
        # x = np.swapaxes(x, 0, 1)

        SOS_TOKEN = torch.full((1, x.size(1), x.size(2)), 2, dtype=torch.float32)
        EOS_TOKEN = torch.full((1, x.size(1), x.size(2)), 3, dtype=torch.float32)
        SOS_TOKEN = SOS_TOKEN.to(x.device)
        EOS_TOKEN = EOS_TOKEN.to(x.device)

        # print("SOS shape:", SOS_TOKEN.size())
        # print("EOS shape:", EOS_TOKEN.size())

        x = torch.cat((SOS_TOKEN, x, EOS_TOKEN), dim=0)

        # Frames x Samples x Hidden Size
        # x = np.swapaxes(x, 0, 1)

        x = self.pos_encoder(x)

        # print("X shape after encoder:", x.size())
        # print("Y shape:", y.size())
        # print("Tgt mask shape:", tgt_mask.size())
        transformer_out = self.transformer(x, y, tgt_mask=tgt_mask)
        return transformer_out[-1, :, :]

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

    def forward(self, audio: Tensor, decoder_dir=None, track=None) -> Tensor:
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
        device = X.device
        
        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            img_width = 255
            hop_length = img_width//2 + 1
            num_frames = X.size(-1)
            arr = torch.zeros(X.size()).to(device)

            if decoder_dir:
                track_path = os.path.join(decoder_dir, track.name)
                track_path = os.path.join(track_path, target_name + ".wav")
                # print("Track path:", track_path)
                sig, rate = torchaudio.load(track_path)
                sig = torch.as_tensor(sig, dtype=torch.float32, device=device)
                sig = utils.preprocess(sig, track.rate, self.sample_rate)
                sig = self.stft(sig)
                sig = self.complexnorm(sig)
                sig = sig.to(device)

            for i in range(0, num_frames, hop_length):
                # print("Indexing from {} to {}".format(i, i+img_width))
                X_tmp = X[:, :, :, i:(i + img_width)]
                if i + img_width > num_frames:
                    padding = (0, i + img_width - num_frames)
                    X_tmp = F.pad(X_tmp, padding, mode='constant', value=0)
                    Y_hat = self.predict(X_tmp, target_module)
                    arr[..., i:] += Y_hat[..., :num_frames - i]
                    break

                Y_hat = self.predict(X_tmp, target_module)
                arr[..., i:i+img_width] += Y_hat
                
                # loss += torch.nn.functional.mse_loss(Y_hat, Y)
            # print("Last frame", i + hop_length, i + img_width, num_hops)

            # Multiply first window and last extra part of window by 2 to make sure that the entire array is doubled
            arr[..., :hop_length] *= 2
            arr[..., i + hop_length:] *= 2

            # Average out window results
            arr /= 2

            target_spectrogram = arr #target_module(X.detach().clone(), y_input, predict=True)

            spectrograms[..., j] = target_spectrogram
            # loss /= i
            # Y_hat = unmix(X, Y)
            # loss = torch.nn.functional.mse_loss(arr, Y)
            # losses.update(loss.item(), Y.size(1))
            
            # # apply current model to get the source spectrogram
            # y_input = torch.full((1, 1, 512), 2, dtype=torch.float32).to(device)
            # # print(X.size())
            # for _ in range(X.size(-1)):
            #     # tgt_mask = target_module.get_tgt_mask(y_input.size(0)).to(device)
            #     pred = target_module(X.detach().clone(), y_input, transformer_only=True).to(device)
            #     pred = pred.unsqueeze(0)
            #     y_input = torch.cat((y_input, pred), dim=0)
            
            # EOS_TOKEN = torch.full((1, y_input.size(1), y_input.size(2)), 3, dtype=torch.float32).to(device)
            # y_input = torch.cat((y_input, EOS_TOKEN), dim=0)
            
            # target_spectrogram = target_module(X.detach().clone(), y_input, predict=True)
            # spectrograms[..., j] = target_spectrogram

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

    def predict(self, X, model):
        y_input = torch.full((X.size(0), X.size(1), X.size(2), 1), -1, dtype=torch.float32).to(X.device)
        print(X.size(), y_input.size())

        for _ in range(X.size(-1)):
            pred = model(X.detach().clone(), y_input, train=False)[..., -1].unsqueeze(-1).to(X.device)
            print(y_input.size(), pred.size())
            y_input = torch.cat((y_input, pred), dim=-1)

        y_hat = y_input[..., 1:]
        print("Final Y_hat size", y_hat.size())
        print("Final X size", X.size())
        y_hat = F.relu(y_hat) * X

        return y_hat

    def forward2(self, audio: Tensor, decoder_dir=None, track=None) -> Tensor:
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
        device = X.device
        
        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            img_width = 255
            hop_length = img_width//2 + 1
            num_frames = X.size(-1)
            arr = torch.zeros(X.size()).to(device)

            track_path = os.path.join(decoder_dir, track.name)
            track_path = os.path.join(track_path, target_name + ".wav")
            # print("Track path:", track_path)
            sig, rate = torchaudio.load(track_path)
            sig = torch.as_tensor(sig, dtype=torch.float32, device=device)
            sig = utils.preprocess(sig, track.rate, self.sample_rate)
            sig = self.stft(sig)
            sig = self.complexnorm(sig)
            sig = sig.to(device)

            for i in range(0, num_frames, hop_length):                
                # print("Indexing from {} to {}".format(i, i+img_width))
                X_tmp, Y_tmp = X[:, :, :, i:(i + img_width)], sig[:, :, :, i:(i + img_width)]
                if i + img_width > num_frames:
                    padding = (0, i + img_width - num_frames)
                    X_tmp, Y_tmp = F.pad(X_tmp, padding, mode='constant', value=0), F.pad(Y_tmp, padding, mode='constant', value=0)
                    Y_hat = target_module(X_tmp, Y_tmp, predict=True)
                    arr[..., i:] += Y_hat[..., :num_frames - i]
                    break

                Y_hat = target_module(X_tmp.detach().clone(), Y_tmp.detach().clone(), predict=True)
                arr[..., i:i+img_width] += Y_hat
                
                # loss += torch.nn.functional.mse_loss(Y_hat, Y)
            # print("Last frame", i + hop_length, i + img_width, num_hops)

            # Multiply first window and last extra part of window by 2 to make sure that the entire array is doubled
            arr[..., :hop_length] *= 2
            arr[..., i + hop_length:] *= 2

            # Average out window results
            arr /= 2

            target_spectrogram = arr #target_module(X.detach().clone(), y_input, predict=True)
                            
            spectrograms[..., j] = target_spectrogram
            # loss /= i
            # Y_hat = unmix(X, Y)
            # loss = torch.nn.functional.mse_loss(arr, Y)
            # losses.update(loss.item(), Y.size(1))
            
            # # apply current model to get the source spectrogram
            # y_input = torch.full((1, 1, 512), 2, dtype=torch.float32).to(device)
            # # print(X.size())
            # for _ in range(X.size(-1)):
            #     # tgt_mask = target_module.get_tgt_mask(y_input.size(0)).to(device)
            #     pred = target_module(X.detach().clone(), y_input, transformer_only=True).to(device)
            #     pred = pred.unsqueeze(0)
            #     y_input = torch.cat((y_input, pred), dim=0)
            
            # EOS_TOKEN = torch.full((1, y_input.size(1), y_input.size(2)), 3, dtype=torch.float32).to(device)
            # y_input = torch.cat((y_input, EOS_TOKEN), dim=0)
            
            # target_spectrogram = target_module(X.detach().clone(), y_input, predict=True)
            # spectrograms[..., j] = target_spectrogram

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
