import torch, torch.nn as nn
from typing import Union
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

    def __repr__(self):
        return "OverlapAdd(n_fft=%s, hop_length=%s)"%(self.n_fft.item(), self.hop_length.item())
    
    def __init__(self, n_fft: int = 1024, hop_length: int = 128, dim:int = -1) -> None:
        super().__init__()
        self.register_buffer("n_fft", torch.tensor(n_fft))
        self.register_buffer("hop_length", torch.tensor(hop_length))
        self.frames_out = (self.n_fft // self.hop_length - 1).item()
        self.register_buffer("input_buffer", torch.zeros(self.frames_out * self.hop_length.item()))
        self.register_buffer("output_buffer", torch.zeros(self.frames_out, self.n_fft.item()))
        self.register_buffer("gain_compensation", torch.tensor(1.))
        self.gain_compensation = self.invert(self._forward_without_update(torch.ones(12, (self.frames_out+1) *self.n_fft))).max()

    def get_input_buffer(self, x: torch.Tensor):
        shape = x.shape
        if self.input_buffer.shape[:-1] != shape[:-1]:
            shape = shape[:-1]
            shape = shape + torch.Size([int(self.frames_out * self.hop_length.item())])
            input_buffer = torch.zeros(shape)
        else:
            input_buffer = self.input_buffer.clone()
        self.input_buffer =  x[..., -(self.frames_out * self.hop_length.item()):]
        return input_buffer

    def get_output_buffer(self, x: torch.Tensor):
        shape = x.shape
        if self.output_buffer.shape[:-1] != shape[:-2]:
            shape = shape[:-2]
            shape = shape + torch.Size([int(self.frames_out * self.hop_length.item())])
            output_buffer = torch.zeros(shape)
        else:
            output_buffer = self.output_buffer.clone()
        return output_buffer

    def _forward_without_update(self, x: torch.Tensor) -> torch.Tensor:
        return frame(x, self.n_fft.item(), self.hop_length.item(), dim=-1)

    @torch.jit.export
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        input_buffer = self.get_input_buffer(x)
        x = torch.cat([input_buffer, x], dim=-1)
        x_framed = frame(x, self.n_fft.item(), self.hop_length.item(), dim=-1)
        return x_framed
    
    @torch.jit.export
    def forward_with_time(self, x: torch.Tensor, time: torch.Tensor):
        # assert x.size(-1) % self.hop_length == 0, "input dim must be a factor of the hop length"
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
    def invert(self, x: torch.Tensor, inversion_mode: Union[str, None] = None, tolerance: Union[float, None] = None) -> torch.Tensor:
        # assert x.size(-1) % self.hop_length == 0, "input dim must be a factor of the hop length"
        n_fft = self.n_fft.item()
        hop_length = self.hop_length.item()
        overlap = int(n_fft / hop_length)
        # perform overlap_add
        output_buffer = self.get_output_buffer(x)
        out_size = x.shape[:-2] + torch.Size([int((x.size(-2)-1) * hop_length + n_fft)])
        recompose_buffer = torch.cat([output_buffer, torch.zeros(out_size)], -1)
        for i in range(x.size(-2)):
            i_shifted = i
            recompose_buffer[..., i * hop_length : i_shifted * hop_length + n_fft] += x[..., i, :] / (overlap / 2)
        out = recompose_buffer[..., :(overlap * hop_length)]
        self.output_buffer = recompose_buffer[..., (overlap * hop_length):]
        return out / self.gain_compensation
