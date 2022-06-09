import torch
import torchaudio
from typing import Union
from .base import AudioTransform, NotInvertibleError
from .norm import Normalize

__all__ = ["MFCC"]


class MFCC(AudioTransform):
    @property
    def invertible(self):
        return False

    @property
    def scriptable(self):
        return True

    @property
    def needs_scaling(self):
        return self.norm is not None

    def __repr__(self):
        _repr_string = f"MFCC(n_fft={self.n_fft}, hop_length={self.hop_length}"\
               f"power={self.power}, n_mels={self.n_mels}"
        if self.norm is not None:
            _repr_string += f", f{self.norm}"
        _repr_string += ")"
        return _repr_string

    def __init__(self, n_fft: int = 1024, hop_length=256, power: float = 2., n_mels: int = 128, sr=44100, norm_mode: str = None):
        super().__init__(sr=sr)
        self.norm: Union[None, Normalize] = None
        if norm_mode is not None:
            self.norm = Normalize(mode=norm_mode)
        self.set_transform(n_fft, n_mels, hop_length, power)

    def set_transform(self, n_fft, n_mels, hop_length, power):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.n_mels = n_mels
        self.transform = torchaudio.transforms.MelSpectrogram(
            self.sr, n_fft, hop_length=hop_length, n_mels=n_mels, power=power)
    
    def forward_with_time(self, x: torch.Tensor, time: torch.Tensor): 
        transform = self.forward(x)
        n_chunks = transform.size(-2)
        shifts = torch.arange(n_chunks) * self.hop_length / self.sr
        shifts = shifts.as_strided((*transform.shape[:-2], n_chunks), (*((0,)*(x.ndim-1)), 1))
        new_time = shifts + time.unsqueeze(-1)
        return transform, new_time

    @torch.jit.export
    def scale_data(self, x: torch.Tensor):
        if self.norm is not None:
            self.norm.scale_data(x)

    @torch.jit.export
    def forward(self, x: torch.Tensor):
        x_t = self.transform(x)
        if self.norm is not None:
            x_t = self.norm(x_t)
        return x_t

    @torch.jit.export
    def invert(self, x: torch.Tensor):
        raise NotInvertibleError
