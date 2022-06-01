import torch
import torch.nn as nn
import enum
import random
from typing import Union, List, Tuple
from .base import AudioTransform
from ..utils.misc import frame


__all__ = ["NormalizeMode", "Normalize"]


class NormalizeMode(enum.Enum):
    UNIPOLAR = 1
    BIPOLAR = 2
    GAUSSIAN = 3


class Normalize(AudioTransform):
    realtime = True
    scriptable = True

    def __repr__(self):
        return f"Normalize(mode={self.mode.name})"

    def __init__(self, mode: Union[NormalizeMode, str] = NormalizeMode.GAUSSIAN):
        super().__init__()
        if isinstance(mode, str):
            mode = NormalizeMode(mode)
        assert mode in NormalizeMode
        self.mode = mode
        self.needs_scaling = True
        self.register_buffer("offset", torch.zeros(0), persistent=False)
        self.register_buffer("scale", torch.ones(1), persistent=False)

    @torch.jit.export
    def scale_data(self, x: torch.Tensor) -> None:
        if self.mode == NormalizeMode.UNIPOLAR:
            self.offset = x.min()
            self.scale = (x - x.min()).max()
        elif self.mode == NormalizeMode.BIPOLAR:
            x_min = x.min()
            x_max = x.max()
            self.offset = (x_max + x_min) / 2
            self.scale = x_max - self.offset
        elif self.mode == NormalizeMode.GAUSSIAN:
            self.offset = x.mean()
            self.scale = x.std()
        self.needs_scaling = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.offset) / self.scale

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.offset

    def test_forward(self, x: torch.Tensor, time: torch.Tensor = None):
        x = frame(x, min(256, x.shape[-1]), min(64, x.shape[-1]), -1)
        x_min = x.min()
        x_max = x.max()
        # test normal mode
        tolerance = torch.finfo(x.dtype).eps
        for norm_mode in NormalizeMode:
            self.mode = norm_mode
            self.scale_data(x)
            x_norm = self(x)
            if norm_mode == NormalizeMode.UNIPOLAR:
                assert x_norm.min() == 0.
                assert x_norm.max() == 1.
            elif norm_mode == NormalizeMode.BIPOLAR:
                assert x_norm.min() == -1.
                assert x_norm.max() == 1.
            elif norm_mode == NormalizeMode.GAUSSIAN:
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
        for norm_mode in NormalizeMode:
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
