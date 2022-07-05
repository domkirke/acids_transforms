import torch
from torchaudio.functional import griffinlim
import math
from .base import AudioTransform, InversionEnumType
from typing import Dict, Union
from ..utils.misc import *

__all__ = ['STFT', 'RealtimeSTFT']

MAX_NFFT = 16384
eps = torch.finfo(torch.get_default_dtype()).eps


class STFT(AudioTransform):
    @property
    def scriptable(self):
        return True

    @property
    def invertible(self):
        return True

    @property
    def needs_scaling(self):
        return False

    def __repr__(self):
        repr_str = f"STFT(n_fft={self.n_fft.item()}, hop_length={self.hop_length.item()}, " \
                   f"inversion_mode = {self.inversion_mode})"
        return repr_str

    def __init__(self,
                 sr: int = 44100,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 dtype: torch.dtype = None,
                 inversion_mode: str = "griffin_lim",
                 window: str = "hann"):
        super().__init__(sr=sr)
        dtype = dtype or torch.get_default_dtype()
        self.register_buffer("n_fft", torch.zeros(1).long())
        self.register_buffer("hop_length", torch.zeros(1).long())
        self.register_buffer("window", torch.zeros(MAX_NFFT))
        self.register_buffer("inv_window", torch.zeros(MAX_NFFT))
        self.register_buffer("gamma", torch.zeros(1))
        self.register_buffer("eps",  torch.tensor(
            torch.finfo(dtype).eps, dtype=dtype))
        self.register_buffer("phase_buffer", torch.zeros(0))

        # set up window
        if hasattr(torch, f"{window}_window"):
            self.window_type = getattr(torch, f"{window}_window")
        else:
            raise ValueError("Window %s is not known" % window)
        if (n_fft is not None):
            assert hop_length is not None, "n_fft and hop_length must be given together"
        if (hop_length is not None):
            assert n_fft is not None, "n_fft and hop_length must be given together"
        # set up parameters
        if (n_fft is not None) and (hop_length is not None):
            self.set_params(n_fft, hop_length)
        if inversion_mode in type(self).get_inversion_modes():
            self.inversion_mode = inversion_mode
        else:
            raise ValueError("Inversion mode %s not known" % inversion_mode)

    @torch.jit.export
    def set_params(self, n_fft: int, hop_length: int) -> None:
        self.n_fft.fill_(n_fft)
        self.hop_length.fill_(hop_length)
        self.window.zero_()
        self.inv_window.zero_()
        self.window[:self.n_fft.item()] = self._get_window()
        self.inv_window[:self.n_fft.item()] = self._get_dual_window()
        self.gamma = self._get_gamma()

    def _get_gamma(self) -> torch.Tensor:
        return 2*torch.pi*((-self.n_fft**2/(8*math.log(0.01)))**.5)**2

    def _get_window(self) -> torch.FloatTensor:
        return self.window_type(self.n_fft.item())

    def _get_dual_window(self) -> torch.Tensor:
        return self._get_window()

    @property
    def ratio(self):
        return self.hop_length.item()

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, batch_shape = reshape_batches(x, -1)
        window = self.window[:self.n_fft.item()]
        x_fft = torch.stft(x, n_fft=self.n_fft.item(), hop_length=self.hop_length.item(
        ), window=window, return_complex=True, onesided=True).transpose(-2, -1)
        self._replace_phase_buffer(x_fft.angle())
        return x_fft.reshape(batch_shape + x_fft.shape[-2:])

    @torch.jit.export
    def forward_with_time(self, x: torch.Tensor, time: torch.Tensor):
        transform = self.forward(x)
        n_chunks = transform.size(-2)
        shifts = torch.arange(n_chunks) * self.hop_length / self.sr
        new_shape = [t for t in transform.shape[:-2]]
        new_shape.append(n_chunks)
        new_strides = [0] * (x.ndim-1)
        new_strides.append(1)
        shifts = shifts.as_strided(new_shape, new_strides)
        new_time = shifts + time.unsqueeze(-1)
        return transform, new_time

    @torch.jit.export
    def invert(self, x: torch.Tensor, inversion_mode: InversionEnumType = None) -> torch.Tensor:
        x, batch_shape = reshape_batches(x, -2)
        if not torch.is_complex(x):
            x_inv = self.invert_without_phase(x, inversion_mode)
        else:
            window = self.inv_window[:self.n_fft.item()]
            x_inv = torch.istft(x.transpose(-2, -1), n_fft=self.n_fft.item(),
                                hop_length=self.hop_length.item(), window=window, onesided=True)
        return x_inv.reshape(batch_shape + x_inv.shape[-1:])

    @staticmethod
    def get_inversion_modes():
        return ['griffin_lim', 'keep_input', 'random']

    def _replace_phase_buffer(self, phase: torch.Tensor) -> None:
        self.phase_buffer = phase

    def _get_phase_buffer(self) -> torch.Tensor:
        phase_buffer = self.phase_buffer
        self.phase_buffer = torch.zeros(0)
        return phase_buffer

    def realtime(self):
        inversion_mode = self.inversion_mode if self.inversion_mode in RealtimeSTFT.get_inversion_modes() else "random"
        return RealtimeSTFT(sr=self.sr, n_fft=self.n_fft.item(), hop_length=self.hop_length.item(), inversion_mode=inversion_mode)

    def invert_without_phase(self, x: torch.Tensor, inversion_mode: InversionEnumType = None) -> torch.Tensor:
        window = self.inv_window[:self.n_fft.item()]
        if inversion_mode is None:
            inversion_mode = self.inversion_mode
        if (inversion_mode == "keep_input"):
            phase = self._get_phase_buffer()
            if phase.shape[0] == 0:
                phase = torch.pi * 2 * torch.rand_like(x)
                x = x * torch.exp(phase * 1j)
                return torch.istft(x.transpose(-2, -1), n_fft=self.n_fft.item(), hop_length=self.hop_length.item(), window=window, onesided=True)
            else:
                x = x * torch.exp(phase * 1j)
                return torch.istft(x.transpose(-2, -1), n_fft=self.n_fft.item(), hop_length=self.hop_length.item(), window=window, onesided=True)
        if (inversion_mode == "griffin_lim"):
            x_inv = self.griffin_lim(x)
            return x_inv
        elif (inversion_mode == "random"):
            phase = torch.pi * 2 * torch.rand_like(x)
            x = x * torch.exp(phase * 1j)
            return torch.istft(x.transpose(-2, -1), n_fft=self.n_fft.item(), hop_length=self.hop_length.item(), window=window, onesided=True)
        else:
            raise ValueError("inversion mode %s not valid." % inversion_mode)

    def griffin_lim(self, x: torch.Tensor) -> torch.Tensor:
        n_fft = self.n_fft.item()
        hop_length = self.hop_length.item()
        window = self.inv_window[:n_fft]
        return griffinlim(x.transpose(-2, -1), window, n_fft, hop_length, n_fft, 1.0, 30, 0.99, None, True)

    # TESTS
    def test_inversion(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outs = {}
        x_stft = self.forward(x)
        outs['direct'] = self.invert(x_stft)
        for inv_type in self.get_inversion_modes():
            outs[inv_type] = self.invert(
                x_stft.abs(), inversion_mode=inv_type)
        return outs

    @classmethod
    def test_scripted_transform(cls, transform: AudioTransform, invert: bool = True, batch_size = (2, 2)):
        x = torch.zeros(*batch_size, 44100)
        time = torch.zeros(*batch_size)
        x_t = transform.forward(x)
        x_t, time_t = transform.forward_with_time(x, time)
        if invert:
            x_inv = transform.invert(x_t)
            for inv_type in cls.get_inversion_modes():
                x_inv = transform.invert(x_t.abs(), inversion_mode=inv_type)


class RealtimeSTFT(STFT):

    def __init__(self, sr: int = 44100, n_fft: int = 1024, hop_length: int = 256, dtype: torch.dtype = None, inversion_mode: InversionEnumType = "random", window: str = "hann", batch_size: int = 2):
        super().__init__(sr=sr, n_fft=n_fft, hop_length=hop_length,
                         dtype=dtype, inversion_mode=inversion_mode, window=window)
        self.batch_size = batch_size

    @property
    def scriptable(self):
        return True

    @property
    def invertible(self):
        return True

    @property
    def needs_scaling(self):
        return False

    def __repr__(self):
        repr_str = f"RealtimeSTFT(n_fft={self.n_fft.item()}, hop_length={self.hop_length.item()}, " \
                   f"inversion_mode = {self.inversion_mode})"
        return repr_str

    @staticmethod
    def get_inversion_modes():
        return ['keep_input', 'random']

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        window = self.window[:self.n_fft.item()]
        x_fft = torch.fft.rfft(x * window)
        self._replace_phase_buffer(x_fft.angle())
        return x_fft
    
    @torch.jit.export
    def forward_with_time(self, x: torch.Tensor, time: torch.Tensor):
        return self(x), time

    @torch.jit.export
    def invert(self, x: torch.Tensor, inversion_mode: InversionEnumType = None) -> torch.Tensor:
        if not torch.is_complex(x):
            x_rec = self.invert_without_phase(x, inversion_mode)
            return x_rec
        else:
            inv_window = self.inv_window[:self.n_fft.item()]
            return torch.fft.irfft(x, norm="backward") * inv_window

    @torch.jit.export
    def get_batch_size(self, batch_size: int):
        return batch_size

    @torch.jit.export
    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def invert_without_phase(self, x: torch.Tensor, inversion_mode: InversionEnumType = None) -> torch.Tensor:
        window = self.inv_window[:self.n_fft.item()]
        if inversion_mode is None:
            inversion_mode = self.inversion_mode
        if (inversion_mode == "keep_input"):
            phase = self._get_phase_buffer()
            if phase.shape[0] == 0:
                phase = torch.pi * 2 * torch.rand_like(x)
        elif (inversion_mode == "random"):
            phase = torch.pi * 2 * torch.rand_like(x)
        else:
            raise ValueError("inversion mode %s not valid." %
                    self.inversion_mode)
        x = x * torch.exp(phase * torch.full(phase.shape,
                                             1j, device=phase.device))
        return torch.fft.irfft(x) * window

    # TESTS
    def test_forward(self, x: torch.Tensor, time: torch.Tensor = None):
        x = frame(x, self.n_fft.item(), self.hop_length.item(), -1)
        transform = []
        for i in range(x.shape[-2]):
            transform.append(self(x[..., i, :]))
        transform = torch.stack(transform, -2)
        if time is None:
            return transform
        else:
            return transform, None

    def test_inversion(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        n_fft = self.n_fft.item()
        hop_length = self.hop_length.item()
        x_framed = frame(x, n_fft, hop_length, -1)
        x_inv_shape = x_framed.size(-2) * hop_length + n_fft
        outs = {k: torch.zeros(*x.shape[:-1], x_inv_shape, device=x.device) for k in [
            'direct'] + self.get_inversion_modes()}
        for n in range(x_framed.size(-2)):
            x_stft = self.forward(x_framed[..., n, :])
            outs['direct'][..., n*hop_length:n *
                           hop_length+n_fft] += self.invert(x_stft)
            for inv_type in self.get_inversion_modes():
                outs[inv_type][..., n*hop_length:n*hop_length +
                               n_fft] += self.invert(x_stft.abs(), inversion_mode=inv_type)
        return outs


    @classmethod
    def test_scripted_transform(cls, transform: AudioTransform, invert: bool = True):
        x = torch.zeros(2, transform.n_fft.item())
        x_t = transform(x)
        if invert:
            x_inv = transform.invert(x_t)
            for inv_type in cls.get_inversion_modes():
                x_inv = transform.invert(x_t.abs(), inversion_mode=inv_type)
