import torch
from typing import Union
from .base import AudioTransform
from ..utils.misc import frame


__all__ = ["Normalize"]

MagnitudeModeType = Union[None, str]


class Normalize(AudioTransform):
    scriptable = True

    def __repr__(self):
        return f"Normalize(mode={self.mode})"

    def __init__(self, mode: MagnitudeModeType = "gaussian"):
        super().__init__()
        self.mode = mode
        self.needs_scaling = True
        self.register_buffer("offset", torch.zeros(0))
        self.register_buffer("scale", torch.ones(1))

    @torch.jit.export
    def scale_data(self, x: torch.Tensor) -> None:
        if self.mode == "unipolar":
            self.offset = x.min()
            self.scale = (x - x.min()).max()
        elif self.mode == "bipolar":
            x_min = x.min()
            x_max = x.max()
            self.offset = (x_max + x_min) / 2
            self.scale = x_max - self.offset
        elif self.mode == "gaussian":
            self.offset = x.mean()
            self.scale = x.std()
        self.needs_scaling = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.offset) / self.scale

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.offset

    def get_normalization_modes(self):
        return ['unipolar', 'bipolar', 'gaussian']

    def test_forward(self, x: torch.Tensor, time: torch.Tensor = None):
        x = frame(x, min(256, x.shape[-1]), min(64, x.shape[-1]), -1)
        x_min = x.min()
        x_max = x.max()
        # test normal mode
        tolerance = torch.finfo(x.dtype).eps
        for norm_mode in self.get_normalization_modes():
            self.mode = norm_mode
            self.scale_data(x)
            x_norm = self(x)
            if norm_mode == "unipolar":
                assert x_norm.min() == 0.
                assert x_norm.max() == 1.
            elif norm_mode == "bipolar":
                assert x_norm.min() == -1.
                assert x_norm.max() == 1.
            elif norm_mode == "gaussian":
                assert (x_norm.mean() < tolerance).item()
                assert ((x_norm.std() - 1).pow(2) < tolerance).item()
            else:
                raise ValueError(
                    "test for norm type %s not implemented" % norm_mode)
        if time is None:
            return x_norm
        else:
            return x_norm, time

    def test_inversion(self, x: torch.Tensor, tolerance: float = None):
        x = frame(x, min(256, x.shape[-1]), min(64, x.shape[-1]), -1)
        x_min = x.min()
        x_max = x.max()
        if tolerance is None:
            tolerance = torch.finfo(x.dtype).eps
        # test normal mode
        for norm_mode in self.get_normalization_modes():
            self.mode = norm_mode
            self.scale_data(x)
            x_norm = self(x)
            x_denorm = self.invert(x_norm)
            assert ((x_min - x_denorm.min()).pow(2) < tolerance).item()
            assert ((x_max - x_denorm.max()).pow(2) < tolerance).item()
        return {}

    @classmethod
    def test_scripted_transform(cls, transform: AudioTransform, invert: bool = True):
        # generate data with random scale & bias
        x = torch.rand((5, 256))
        transform.scale_data(x)
        x_normalized = transform(x)
        if invert:
            x_inv = transform(x_normalized)
