import torch, torch.nn as nn

from .base import AudioTransform
from ..utils import frame

class OverlapAdd(AudioTransform):

    @property
    def invertible(self):
        return True
        
    @property
    def scriptable(self):
        return True

    @property
    def needs_scaling(self):
        return False
    
    def __init__(self, n_fft: int = 2048, hop_length: int = 256, dim:int = -1) -> None:
        super().__init__()
        self.register_buffer("n_fft", torch.tensor(n_fft))
        self.register_buffer("hop_length", torch.tensor(hop_length))
        frames_out = (self.n_fft // self.hop_length - 1).item()
        self.register_buffer("causal_buffer", torch.zeros(frames_out * self.hop_length.item()))
        self.register_buffer("gain_compensation", torch.tensor(1.))
        self.gain_compensation = self.invert(self(torch.ones(12, self.n_fft), update=False)).max()

    def get_causal_buffer(self, x: torch.Tensor):
        shape = x.shape
        frames_out = (self.n_fft // self.hop_length - 1).item()
        if self.causal_buffer.shape != shape:
            shape = shape[:-1]
            shape = shape + torch.Size([int(frames_out * self.hop_length.item())])
            causal_buffer = torch.zeros(shape)
        else:
            causal_buffer = self.causal_buffer.clone()
        self.causal_buffer =  x[..., -(frames_out * self.hop_length.item()):]
        return causal_buffer

    @torch.jit.export
    def forward(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        if update:
            causal_buffer = self.get_causal_buffer(x)
            x = torch.cat([causal_buffer, x], dim=-1)
        x_framed = frame(x, self.n_fft.item(), self.hop_length.item(), dim=-1)
        return x_framed
    
    @torch.jit.export
    def forward_with_time(self, x: torch.Tensor, time: torch.Tensor):
        assert x.size(-1) % self.hop_length == 0, "input dim must be a factor of the hop length"
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
    def invert(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) % self.hop_length == 0, "input dim must be a factor of the hop length"
        n_fft = self.n_fft.item()
        hop_length = self.hop_length.item()
        frames_out = int(n_fft / hop_length) - 1
        overlap = int(n_fft / hop_length)
        # perform overlap_add
        out_size = x.shape[:-2] + torch.Size([int((x.size(-2)-1) * hop_length + n_fft)])
        out = torch.zeros(out_size)
        for i in range(x.size(-2)):
            out[..., i * hop_length : i* hop_length + n_fft] += x[..., i, :] / (overlap / 2)
        return out[..., frames_out * hop_length: -(frames_out * hop_length)] / self.gain_compensation
