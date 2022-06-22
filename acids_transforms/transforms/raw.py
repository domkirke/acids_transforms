import torch
import torchaudio
from .base import AudioTransform, InversionEnumType
from ..utils.misc import frame
from typing import Dict


__all__ = ['Mono', 'Stereo', 'Window', 'MuLaw']


class Mono(AudioTransform):
    @property
    def scriptable(self):
        return True

    @property
    def invertible(self):
        return True

    @property
    def needs_scaling(self):
        return False

    def __init__(self, mode: str = "mix", normalize: bool = False, squeeze: bool = True, inversion_mode="mono"):
        super().__init__()
        self.mode = mode
        self.squeeze = squeeze
        self.normalize = normalize
        self.inversion_mode = inversion_mode

    def __repr__(self):
        return "Mono(mode=%s, normalize=%s squeeze=%s, inversion_mode=%s)" % (self.mode, self.normalize, self.squeeze, self.inversion_mode)

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, list):
            return [self(x_tmp) for x_tmp in x]
        if x.shape[-2] == 2:
            if self.mode == "mix":
                x = (x.sum(-2) / 2).unsqueeze(-2)
            elif self.mode == "right":
                x = x.index_select(-2, torch.tensor(1))
            elif self.mode == "left":
                x = x.index_select(-2, torch.tensor(0))
        if self.normalize:
            x = x / x.max()
        if self.squeeze:
            x = x.squeeze(-2)
        return x

    @torch.jit.export
    def forward_with_time(self, x: torch.Tensor, time: torch.Tensor):
        if self.squeeze:
            time = time[..., 0]
        else:
            time = time[..., 0].unsqueeze(-1)
        return self(x), time

    def get_inversion_modes(self):
        return ['mono', 'stereo']

    @torch.jit.export
    def invert(self, x, inversion_mode: InversionEnumType = None, tolerance: float = 0.0):
        if inversion_mode is None:
            inversion_mode = self.inversion_mode
        if self.squeeze:
            x = x.unsqueeze(-2)
        if x.shape[-2] == 1 and self.inversion_mode == "stereo":
            x = torch.cat([x, x], dim=-2)
        return x

    def test_inversion(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outs = {}
        x_t = self.forward(x)
        for inv_type in self.get_inversion_modes():
            outs[inv_type] = self.invert(x_t, inversion_mode=inv_type)
        return outs


class Stereo(AudioTransform):
    @property
    def scriptable(self):
        return True

    @property
    def invertible(self):
        return True

    @property
    def needs_scaling(self):
        return False

    def __init__(self, normalize=False, sr=44100):
        super().__init__()
        self.normalize = normalize

    def __repr__(self):
        return "Stereo(normalize=%s)" % self.normalize

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 1:
            x = torch.cat([x, x], dim=0)
        if self.normalize:
            x = x / x.max()
        return x

    @torch.jit.export
    def invert(self, x: torch.Tensor, inversion_mode: InversionEnumType = None, tolerance: float = 1.e-4) -> torch.Tensor:
        if x.shape[0] == 1:
            x = torch.cat([x, x], dim=0)
        return x


class Window(AudioTransform):

    @property
    def scriptable(self):
        return True

    @property
    def invertible(self):
        return True

    @property
    def needs_scaling(self):
        return False

    def __init__(self, sr: int = 44100, window_size: int = 1024, hop_size: int = 256, dim: int = -1, batch_dim: int = 0, inversion_mode: str = "crop"):
        super().__init__()
        self.sr = sr
        self.window_size = window_size
        self.hop_size = hop_size or self.window_size
        assert self.window_size >= self.hop_size
        self.dim = dim
        self.batch_dim = batch_dim
        self.inversion_mode = inversion_mode

    def __repr__(self):
        return f"Window(ws={self.window_size}, hs={self.hop_size}, dim={self.dim}, pad={self.pad}, inversion={self.inversion})"

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = frame(x, self.window_size, self.hop_size, self.dim)
        return chunks

    @property
    def ratio(self):
        return self.hop_size.item()

    @torch.jit.export
    def forward_with_time(self, x: torch.Tensor, time: torch.Tensor):
        transform = self.forward(x)
        n_chunks = transform.size(-2)
        shifts = torch.arange(n_chunks) * self.hop_size / self.sr
        new_shape = [t for t in transform.shape[:-2]]
        new_shape.append(n_chunks)
        new_strides = [0] * (x.ndim-1)
        new_strides.append(1)
        shifts = shifts.as_strided(new_shape, new_strides)
        new_time = shifts + time.unsqueeze(-1)
        return transform, new_time
    # def test_inversion(self, x: torch.Tensor):
    #     x = torch.arange(100).unsqueeze(0).unsqueeze(1)
    #     x = x.repeat(4, 3, 1)
    #     w = Window(10, 5)
    #     x_w = w(x)
    #     y = w.invert(x_w)

    @torch.jit.export
    def invert(self, x: torch.Tensor, inversion_mode:  InversionEnumType = None, tolerance: float = 1.e-4) -> torch.Tensor:
        if inversion_mode is None:
            inversion_mode = self.inversion_mode
        if self.dim >= 0:
            dim = self.dim 
        else:
            dim = len(x.shape) + self.dim 

        if self.window_size == self.hop_size:
            old_shape = list(x.shape)
            new_shape = old_shape[:dim-1]
            new_shape.append(old_shape[dim-1]*old_shape[dim])
            new_shape.extend(old_shape[dim+1:])
            x = x.reshape(new_shape)
        else:
            if self.inversion_mode == "crop":
                new_x = x.index_select(self.dim, torch.arange(self.hop_size))
                new_x = list(new_x.split(1, self.dim-1))
                x_tail = x.index_select(dim-1, torch.tensor([x.size(dim-1)-1]).long())
                x_tail = x_tail.index_select(dim, torch.arange(self.hop_size, x.size(dim)))
                new_x.append(x_tail)
                # concatenate
                x = torch.cat(new_x, dim).squeeze(dim-1)
        return x


class MuLaw(AudioTransform):
    @property
    def scriptable(self):
        return True

    @property
    def invertible(self):
        return True

    @property
    def needs_scaling(self):
        return False

    def __init__(self, channels=256, one_hot="none", **kwargs):
        super().__init__()
        self.channels = channels
        self.one_hot = one_hot
        self.encoding = torchaudio.transforms.MuLawEncoding(channels)
        self.decoding = torchaudio.transforms.MuLawDecoding(channels)

    def encode(self, x):
        out = self.encoding(x)
        if self.one_hot == "channel":
            out = torch.nn.functional.one_hot(
                out, self.channels).transpose(-1, -2).contiguous()
        elif self.one_hot == "categorical":
            out = torch.nn.functional.one_hot(out, self.channels)
        return out

    def decode(self, x):
        x = x.long()
        if self.one_hot == "channel":
            x = x.transpose(-2, -1)
            batch_shape = x.shape[:-2]
            idx = x.view(-1, x.shape[-2], x.shape[-1]).nonzero()[:, -1]
            out = idx.reshape(*batch_shape, -1)
        elif self.one_hot == "categorical":
            batch_shape = x.shape[:-2]
            idx = x.view(-1, x.shape[-2], x.shape[-1]).nonzero()[:, 1]
            out = idx.reshape(*batch_shape, -1)
        else:
            out = x
        out = self.decoding(out)
        return out

    @torch.jit.export
    def forward(self, x):
        return self.encode(x)

    @torch.jit.export
    def invert(self, x, inversion_mode: InversionEnumType = None, tolerance: float = 1.e-4):
        return self.decoding(x)
