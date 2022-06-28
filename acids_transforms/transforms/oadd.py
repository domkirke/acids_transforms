import torch, torch.nn as nn
from ..utils import frame

class OverlapAdd(nn.Module):
    def __init__(self, n_fft, hop_length, dim:int = -1) -> None:
        super().__init__()
        self.register_buffer("n_fft", torch.tensor(n_fft))
        self.register_buffer("hop_length", torch.tensor(hop_length))
        frames_out = (self.n_fft // self.hop_length - 1).item()
        self.register_buffer("causal_buffer", torch.zeros(frames_out * self.hop_length.item()))

    def get_causal_buffer(self, x: torch.Tensor):
        shape = x.shape
        frames_out = (self.n_fft // self.hop_length - 1).item()
        if self.causal_buffer.shape != shape:
            shape = shape[:-1]
            shape = shape + torch.Size([int(frames_out * self.hop_length.item())])
            causal_buffer = torch.zeros(shape)
        else:
            causal_buffer = self.causal_buffer.clone()
        self.causal_buffer =  x[:, -(frames_out * self.hop_length.item()):]
        return causal_buffer

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        causal_buffer = self.get_causal_buffer(x)
        x = torch.cat([causal_buffer, x], dim=-1)
        x_framed = frame(x, self.n_fft.item(), self.hop_length.item(), dim=-1)
        return x_framed
    
    @torch.jit.export
    def invert(self, x: torch.Tensor) -> torch.Tensor:
        n_fft = self.n_fft.item()
        hop_length = self.hop_length.item()
        frames_out = int(n_fft / hop_length) - 1
        overlap = int(n_fft / hop_length)
        # perform overlap_add
        out_size = x.shape[:-2] + torch.Size([int((x.size(-2)-1) * hop_length + n_fft)])
        out = torch.zeros(out_size)
        for i in range(x.size(-2)):
            out[..., i * hop_length : i* hop_length + n_fft] += x[..., i, :] / (overlap / 2)
        return out[..., frames_out * hop_length: -(frames_out * hop_length)]