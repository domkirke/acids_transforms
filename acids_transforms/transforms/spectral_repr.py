from numbers import Real
from enum import Enum

from numpy import reshape
from .base import AudioTransform, InversionEnumType
from ..utils.misc import *
from .stft import STFT
from .norm import Normalize
from typing import Union, Tuple
import torch
import torch.nn as nn

__all__ = ["Real", "Imaginary", "Magnitude", 
           "Phase", "IF", "Cartesian", "Polar", "PolarIF"]


class Dummy(AudioTransform):
    pass


class _Representation(AudioTransform):

    def __init__(self, sr: int = 44100, mode: Union[str, None] = None, keep_nyquist: bool = True):
        super().__init__(sr=sr)
        if (mode is None) or (mode == "none"):
            self.norm = Dummy()
        else:
            self.norm = Normalize(mode)
        self.keep_nyquist = keep_nyquist

    @property
    def scriptable(self):
        return True

    @property
    def invertible(self):
        return True

    @property
    def needs_scaling(self):
        return True

    @torch.jit.export
    def scale_data(self, x: torch.Tensor) -> None:
        return self.norm.scale_data(x)

    @torch.jit.export
    def invert(self, x, inversion_mode: InversionEnumType = None, tolerance: float = 1.e-4) -> torch.Tensor:
        out = self.norm.invert(x)
        if not self.keep_nyquist:
            pad_size = out.shape[:-1] + torch.Size([1])
            pad_tensor = torch.zeros(pad_size, device=out.device, dtype=out.dtype)
            out = torch.cat([out, pad_tensor], -1)
        return out

    @classmethod
    def test_scripted_transform(cls, transform, invert: bool = True):
        shape = (2, 10, 513)
        complex_random = torch.randn(
            *shape) * torch.exp(2 * torch.pi * torch.rand(*shape) * torch.full(shape, 1j))
        transform.scale_data(complex_random)
        x_repr = transform(complex_random)
        if invert:
            x_inv = transform.invert(x_repr)

    def test_forward(self, x: torch.Tensor, time: torch.Tensor = None):
        stft_transform = STFT()
        if time is None:
            x_fft = stft_transform(x)
            self.scale_data(x_fft)
            return self(x_fft)
        else:
            x_fft, time = stft_transform.forward_with_time(x, time)
            self.scale_data(x_fft)
            return self.forward_with_time(x_fft, time)


class Real(_Representation):

    def __repr__(self):
        return "Real(norm=%s)" % self.norm.mode

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.keep_nyquist:
            x = x[..., 1:]
        return self.norm(x.real)

    @torch.jit.export
    def scale_data(self, x: torch.Tensor) -> None:
        return self.norm.scale_data(x.real)

    def test_inversion(self, x: torch.Tensor):
        # simulate STFT
        x, batch_size = reshape_batches(x, -1)
        x = torch.stft(x, 512, 128, return_complex=True)
        imag = x.imag
        self.scale_data(x)
        x_real = self(x)
        x_real_inv = self.invert(x_real)
        x_inv_fft = x_real_inv + imag * 1j
        x_inv = torch.istft(x_inv_fft, 512, 128)
        x_inv = x_inv.reshape(*batch_size, x_inv.shape[-1])
        return {'direct': x_inv}


class Imaginary(_Representation):

    def __repr__(self):
        return "Imaginary(norm=%s)" % self.norm.mode

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(x):
            x = self.norm(x.imag)
        else:
            x =  torch.zeros_like(x)
        if not self.keep_nyquist:
            x = x[..., 1:]
        return x

    @torch.jit.export
    def scale_data(self, x: torch.Tensor) -> None:
        return self.norm.scale_data(x.imag)

    def test_inversion(self, x: torch.Tensor):
        # simulate STFT
        x, batch_size = reshape_batches(x, -1)
        x = torch.stft(x, 1024, 256, return_complex=True).transpose(-2, -1)
        real = x.real
        self.scale_data(x)
        x_imag = self(x)
        x_imag_inv = self.invert(x_imag)
        x_inv_fft = real + x_imag_inv * 1j
        x_inv = torch.istft(x_inv_fft.transpose(-2, -1), 1024, 256)
        x_inv = x_inv.reshape(*batch_size, x_inv.shape[-1])
        return {'direct': x_inv}


ContrastModeType = Union[None, str]


class Magnitude(_Representation):

    def __repr__(self):
        if self.mel:
            return "Magnitude(mel=%s, n_fft=%s, norm=%s)" % (self.mel, self.n_fft, self.norm.mode)
        else:
            return "Magnitude(norm=%s)" % self.norm.mode

    def __init__(self,
                 sr: int = 44100,
                 mode: Union[str, None] = "unipolar",
                 contrast: ContrastModeType = "log1p",
                 mel: bool = True,
                 n_fft: int = 1024,
                 dtype: torch.dtype = None,
                 eps: float = None,
                 keep_nyquist: bool = True):
        super().__init__(sr=sr, mode=mode)
        self.contrast_mode = contrast
        self.mel = mel
        self.keep_nyquist = True
        if dtype is None:
            dtype = torch.get_default_dtype()
        if eps is None:
            eps = torch.finfo(dtype).eps
        self.register_buffer("eps", torch.tensor(eps))
        self.keep_nyquist = keep_nyquist
        # make mel bank
        assert sr is not None
        assert n_fft is not None
        n_bins = n_fft // 2 + 1
        fft_scale = torch.arange(n_fft // 2 + 1) / n_fft * sr
        if not self.keep_nyquist:
            fft_scale = fft_scale[..., 1:]
        mel_bank = torchaudio.functional.melscale_fbanks(
            n_bins, fft_scale[0], fft_scale[-1], n_bins, sr)
        mel_norm = mel_bank.sum(0)
        mel_bank_norm = mel_bank / \
            torch.where(mel_norm != 0, mel_norm,
                        torch.ones_like(mel_norm)).unsqueeze(0)
        mel_inv_norm = mel_bank.sum(1)
        mel_bank_inv_norm = mel_bank / \
            torch.where(mel_inv_norm != 0, mel_inv_norm,
                        torch.ones_like(mel_inv_norm)).unsqueeze(1)
        self.register_buffer("mel_bank", mel_bank_norm.unsqueeze(0))
        self.register_buffer(
            "inverse_mel_bank", mel_bank_inv_norm.transpose(-2, -1).unsqueeze(0))

    def contrast(self, mag: torch.Tensor) -> torch.Tensor:
        if self.contrast_mode == "log1p":
            return torch.log(1 + mag)
        elif self.contrast_mode == "log":
            return torch.log(torch.clamp(mag, self.eps, None))
        elif self.contrast_mode == "log10":
            return torch.log10(torch.clamp(mag, self.eps, None))
        elif (self.contrast_mode is None) or (self.contrast_mode == "none"):
            return mag
        else:
            raise TypeError("unknown contrast type %s" % self.contrast_mode)

    def invert_contrast(self, mag: torch.Tensor) -> torch.Tensor:
        if self.contrast_mode == "log1p":
            return torch.exp(mag) - 1
        elif self.contrast_mode == "log":
            return torch.exp(mag) - self.eps
        elif self.contrast_mode == "log10":
            return torch.tensor(10).pow(mag)
        elif (self.contrast_mode is None) or (self.contrast_mode == "none"):
            return mag
        else:
            raise TypeError("unknown contrast type %s" % self.contrast_mode)

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = x.abs()
        if self.mel:
            mag = torch.matmul(mag, self.mel_bank)
            if x.ndim <=2:
                mag = mag[0]
        mag = self.contrast(mag)
        mag = self.norm(mag)
        if not self.keep_nyquist:
            mag = mag[..., 1:]
        return mag

    @torch.jit.export
    def invert(self, x: torch.Tensor, inversion_mode: InversionEnumType = None, tolerance: float = 1.e-4) -> torch.Tensor:
        mag = self.norm.invert(x)
        if not self.keep_nyquist:
            pad_size = mag.shape[:-1] + torch.Size([1])
            pad_tensor = torch.zeros(pad_size, device=mag.device, dtype=mag.dtype)
            mag = torch.cat([mag, pad_tensor], -1)
        mag = self.invert_contrast(mag)
        if self.mel:
            mag = torch.matmul(mag, self.inverse_mel_bank)
            if x.ndim <= 2:
                mag = mag[0]
        return mag

    @torch.jit.export
    def scale_data(self, x: torch.Tensor) -> None:
        x = self.contrast(x.abs())
        return self.norm.scale_data(x)

    def test_inversion(self, x: torch.Tensor):
        # simulate STFT
        x, batch_size = reshape_batches(x, -1)
        x = torch.stft(x, 1024, 256, return_complex=True).transpose(-2, -1)
        angle = x.angle()
        self.scale_data(x)
        x_mag = self(x)
        x_mag_inv = self.invert(x_mag)
        x_inv_fft = x_mag_inv * torch.exp(angle * 1j)
        x_inv = torch.istft(x_inv_fft.transpose(-2, -1), 1024, 256)
        x_inv = x_inv.reshape(*batch_size, x_inv.shape[-1])
        return {'direct': x_inv}


class Phase(_Representation):

    def __init__(self, sr: int = 44100, mode: Union[str, None] = None, keep_nyquist: bool = True, unwrap: bool = False):
        super(Phase, self).__init__(sr=sr, mode=mode, keep_nyquist=keep_nyquist)
        self.unwrap = unwrap

    def __repr__(self):
        return "Phase(norm=%s, unwrap=%s)" % (self.norm.mode, self.unwrap)

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.angle()
        if self.unwrap:
            x = unwrap(x)
        x = self.norm(x)
        if not self.keep_nyquist:
            x = x[..., 1:]
        return x

    @torch.jit.export
    def scale_data(self, x: torch.Tensor) -> None:
        x = x.angle()
        if self.unwrap:
            x = unwrap(x)
        return self.norm.scale_data(x)

    def test_inversion(self, x: torch.Tensor):
        # simulate STFT
        x, batch_size = reshape_batches(x, -1)
        x = torch.stft(x, 1024, 256, return_complex=True)
        mag = x.abs()
        self.scale_data(x)
        x_angle = self(x)
        x_angle_inv = self.invert(x_angle)
        x_inv_fft = mag * torch.exp(x_angle_inv * 1j)
        x_inv = torch.istft(x_inv_fft, 1024, 256)
        x_inv = x_inv.reshape(*batch_size, x_inv.shape[-1])
        return {'direct': x_inv}



class IF(_Representation):

    def __repr__(self):
        return "IF(method=%s, norm=%s)" % (self.method, self.norm.mode)

    def __init__(self, sr: int = 44100, mode: Union[str, None] = "gaussian", method: Union[str, None] = "forward", weighted=False, keep_nyquist:bool = True):
        super().__init__(sr=sr, mode=mode)
        self.method = method
        self.weighted = weighted
        self.weighted_window = torch.zeros(0, 0)
        self.keep_nyquist = keep_nyquist
        self.register_buffer("eps", torch.tensor(
            torch.finfo(torch.float32).eps))

    def get_if_methods(self):
        return ["backward", "forward", "central"]

    def get_if(self, data: torch.Tensor):
        phase = unwrap(data.angle())
        if self.method == "backward":
            inst_f = fdiff_backward(phase)
            inst_f[..., 1:, :] /= (-torch.pi)
        elif self.method == "forward":
            inst_f = fdiff_forward(phase)
            inst_f[..., :-1, :] /= (torch.pi)
        elif self.method == "central":
            inst_f = fdiff_central(phase)
            inst_f[..., 1:-1, :] /= (2 * torch.pi)
        else:
            raise AttributeError("method %s not known" % self.method)
        if self.weighted:
            window = self._get_weighted_window(inst_f)
            inst_f = window * inst_f
        return inst_f

    def _get_weighted_window(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(-2)
        if self.weighted_window.size(-2) != x.size(-2):
            n = torch.arange(N)
            self.weighted_window = (
                1.5 * N) / (N ** 2 - 1) * (1 - ((n - (N / 2 - 1)) / (N / 2)) ** 2)
        window_shape = [1] * x.ndim
        window_shape[-2] = N
        return self.weighted_window.view(window_shape)

    @torch.jit.export
    def scale_data(self, x: torch.Tensor):
        self.norm.scale_data(self.get_if(x))

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inst_f = self.get_if(x)
        inst_f = self.norm(inst_f)
        if not self.keep_nyquist:
            inst_f = inst_f[..., 1:]
        return inst_f

    @torch.jit.export
    def invert(self, x, inversion_mode: InversionEnumType = None, tolerance: float = 1.e-4) -> torch.Tensor:
        x_denorm = self.norm.invert(x)
        if self.method == "backward":
            x_denorm[..., 1:, :] *= -torch.pi
            x_denorm = fint_backward(x_denorm)
        if self.method == "forward":
            x_denorm[..., :-1, :] *= torch.pi
            x_denorm=fint_forward(x_denorm)
        elif self.method == "central":
            x_denorm[..., 1:-1, :] *= (2 * torch.pi)
            x_denorm=fint_central(x_denorm)
        if not self.keep_nyquist:
            pad_size = x_denorm.shape[:-1] + torch.Size([1])
            pad_tensor = torch.zeros(pad_size, device=x_denorm.device, dtype=x_denorm.dtype)
            x_denorm = torch.cat([x_denorm, pad_tensor], -1)
        return x_denorm

    def test_inversion(self, x: torch.Tensor):
        # simulate STFT
        x, batch_size = reshape_batches(x, -1)
        x=torch.stft(x, 1024, 256, return_complex=True).transpose(-2, -1)
        mag=x.abs()
        outs = {}
        for inv_mode in self.get_if_methods():
            self.method = inv_mode 
            self.scale_data(x)
            x_angle=self(x)
            x_angle_inv=self.invert(x_angle)
            x_inv_fft=mag * torch.exp(x_angle_inv * 1j)
            x_inv=torch.istft(x_inv_fft.transpose(-2, -1), 1024, 256)
            x_inv = x_inv.reshape(*batch_size, x_inv.shape[-1])
            outs[inv_mode] = x_inv
        return outs


SpectralRepresentationType=Union[torch.Tensor,
                                   Tuple[torch.Tensor, torch.Tensor]]


class SpectralRepresentation(AudioTransform):

    @ property
    def scriptable(self):
        return True

    @ property
    def invertible(self):
        return True

    @ property
    def needs_scaling(self):
        return True

    def __init__(self, sr: int=44100,
                 magnitude_transform=None, phase_transform=None,
                 magnitude_args={}, phase_args={}, stack=-2,
                 keep_nyquist: bool = True):
        super().__init__(sr=sr)
        if type(self) == SpectralRepresentation:
            raise RuntimeError(
                "SpectralRepresentation should not be called directly.")
        self.keep_nyquist=keep_nyquist
        self.magnitude=magnitude_transform(sr=sr, **magnitude_args, keep_nyquist=keep_nyquist)
        self.phase=phase_transform(sr=sr, **phase_args, keep_nyquist=keep_nyquist)
        self.stack=stack

    @ torch.jit.export
    def scale_data(self, x: torch.Tensor) -> None:
        self.magnitude.scale_data(x)
        self.phase.scale_data(x)

    @ torch.jit.export
    def forward(self, x: torch.Tensor) -> SpectralRepresentationType:
        magnitude=self.magnitude(x)
        phase=self.phase(x)
        if self.stack is not None:
            return torch.stack([magnitude, phase], dim=self.stack)
        else:
            return (magnitude, phase)

    @ torch.jit.export
    def invert(self, x, inversion_mode: InversionEnumType=None, tolerance: float=1.e-4) -> torch.Tensor:
        if self.stack is None:
            mag=x[0]
            phase=x[1]
        else:
            mag=x.index_select(
                self.stack, torch.tensor(0)).squeeze(self.stack)
            phase=x.index_select(
                self.stack, torch.tensor(1)).squeeze(self.stack)
        mag=self.magnitude.invert(mag)
        phase=self.phase.invert(phase)
        return mag * torch.exp(phase * torch.full(phase.shape, 1j, device=phase.device))

    def test_forward(self, x: torch.Tensor, time: torch.Tensor=None):
        stft_transform=STFT()
        if time is None:
            x_fft=stft_transform(x)
            self.scale_data(x_fft)
            return self(x_fft)
        else:
            x_fft, time=stft_transform.forward_with_time(x, time)
            self.scale_data(x_fft)
            return self.forward_with_time(x_fft, time)

    def test_inversion(self, x: torch.Tensor):
        # simulate STFT
        x, batch_size = reshape_batches(x, -1)
        x=torch.stft(x, 1024, 256, return_complex=True).transpose(-1, -2)
        self.scale_data(x)
        x_t=self(x)
        x_t_inv=self.invert(x_t)
        x_inv=torch.istft(x_t_inv.transpose(-1, -2), 1024, 256)
        x_inv = x_inv.reshape(*batch_size, x_inv.shape[-1])
        return {'direct': x_inv}

    @ classmethod
    def test_scripted_transform(cls, transform, invert: bool=True):
        complex_random=torch.randn(
            2, 10, 513) * torch.exp(2 * torch.pi * torch.rand(2, 10, 513) * torch.full((2, 10, 513), 1j))
        transform.scale_data(complex_random)
        x_repr=transform(complex_random)
        if invert:
            x_inv=transform.invert(x_repr)


class Cartesian(SpectralRepresentation):

    def __repr__(self):
        return "Cartesian(real_norm=%s, imag_norm=%s)" % (self.magnitude.norm.mode, self.phase.norm.mode)

    def __init__(self, sr: int=44100, real_args={"mode":"gaussian"}, imag_args={"mode":"gaussian"},
                 stack=-2, keep_nyquist: bool = True):
        super(Cartesian, self).__init__(sr, Real, Imaginary,
                                        real_args, imag_args, stack=stack, keep_nyquist=keep_nyquist)

    @ torch.jit.export
    def invert(self, x, inversion_mode: InversionEnumType=None, tolerance: float=1.e-4) -> torch.Tensor:
        if self.stack is None:
            real=x[0]
            imag=x[1]
        else:
            real=x.index_select(
                self.stack, torch.tensor(0)).squeeze(self.stack)
            imag=x.index_select(
                self.stack, torch.tensor(1)).squeeze(self.stack)
        real=self.magnitude.invert(real)
        imag=self.phase.invert(imag)
        return real + imag * torch.full(imag.shape, 1j, device=imag.device)


class Polar(SpectralRepresentation):

    def __repr__(self):
        return "Polar(real_norm=%s, imag_norm=%s)" % (self.magnitude.norm.mode, self.phase.norm.mode)

    def __init__(self,
                 sr: int=44100,
                 magnitude_args = {"mode":"bipolar"},
                 phase_args = {"mode":"bipolar"},
                 stack=-2, keep_nyquist: bool = True):
        super(Polar, self).__init__(sr, Magnitude, Phase,
                                    magnitude_args, phase_args, stack=stack, keep_nyquist=keep_nyquist)


class PolarIF(SpectralRepresentation):

    def __repr__(self):
        return "PolarIF(real_norm=%s, imag_norm=%s)" % (self.magnitude.norm.mode, self.phase.norm.mode)

    def __init__(self,
                 sr: int=44100,
                 magnitude_args = {"mode":"bipolar"},
                 phase_args = {"mode":"bipolar"},
                 stack=-2, 
                 keep_nyquist: bool = True):
        super(PolarIF, self).__init__(sr, Magnitude, IF,
                                      magnitude_args, phase_args, stack=stack, keep_nyquist=keep_nyquist)

    def test_inversion(self, x: torch.Tensor):
        # reshape x in case
        x, batch_size = reshape_batches(x, -1)
        # simulate STFT
        x=torch.stft(x, 1024, 256, return_complex=True).transpose(-1, -2)
        outs={}
        for method in self.phase.get_if_methods():
            self.phase.method=method
            self.scale_data(x)
            x_t=self(x)
            x_t_inv=self.invert(x_t)
            x_inv=torch.istft(x_t_inv.transpose(-1, -2), 1024, 256)
            outs[method]=x_inv.reshape(*batch_size, x_inv.shape[-1])
        return outs
