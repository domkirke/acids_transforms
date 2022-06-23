from turtle import forward
import torch
from .base import AudioTransform, InversionEnumType, NotInvertibleError
from ..utils import frame
from typing import Tuple, Dict, Union


class Unsqueeze(AudioTransform):

    @property
    def scriptable(self):
        return True

    @property
    def invertible(self):
        return self.dim is not None

    @property
    def needs_scaling(self):
        return False

    def __repr__(self):
        return "Unsqueeze(dim=%s)" % self.dim

    def __init__(self, sr=44100, dim=1):
        super().__init__(sr)
        self.dim = dim

    @torch.jit.export
    def forward(self, x: torch.Tensor):
        return x.unsqueeze(self.dim)

    @torch.jit.export
    def invert(self, x: torch.Tensor, inversion_mode: InversionEnumType = None, tolerance: float = 1.e-4) -> torch.Tensor:
        return x.squeeze(self.dim)

    def test_forward(self, x: torch.Tensor, time: torch.Tensor = None):
        fake_size = (2, 512)
        x_fake = torch.zeros(fake_size)
        assert self(x_fake).shape == (2, 1, 512)
        if time is None:
            return x_fake
        else:
            return x_fake, time

    def test_inversion(self, x: torch.Tensor):
        fake_size = (2, 512)
        x_fake = self.forward(torch.zeros(fake_size))
        x_inv = self.invert(x_fake)
        assert x_inv.shape == fake_size
        return {}


class Squeeze(AudioTransform):

    @property
    def scriptable(self):
        return True

    @property
    def invertible(self):
        return self.dim is not None

    @property
    def needs_scaling(self):
        return False

    def __init__(self, sr=44100, dim=None):
        super().__init__(sr)
        self.dim = dim

    def __repr__(self):
        return "Squeeze(dim=%s)" % self.dim

    @torch.jit.export
    def forward(self, x: torch.Tensor):
        if self.dim is None:
            return x.squeeze()
        else:
            return x.squeeze(self.dim)

    @torch.jit.export
    def invert(self, x: torch.Tensor, inversion_mode: InversionEnumType = None, tolerance: float = 1.e-4) -> torch.Tensor:
        if self.dim is None:
            raise NotInvertibleError
        else:
            return x.unsqueeze(self.dim)

    def test_forward(self, x: torch.Tensor, time: torch.Tensor = None):
        # full squeeze
        self.dim = None
        fake_size = (2, 1, 512, 1)
        x_fake = torch.zeros(fake_size)
        assert self(x_fake).shape == (2, 512)
        # partial squeeze
        self.dim = 1
        fake_size = (2, 1, 512, 1)
        x_fake = torch.zeros(fake_size)
        assert self(x_fake).shape == (2, 512, 1)
        if time is None:
            return x_fake
        else:
            return x_fake, time

    def test_inversion(self, x: torch.Tensor):
        self.dim = 1
        fake_size = (2, 1, 512, 1)
        x_fake = self.forward(torch.zeros(fake_size))
        x_inv = self.invert(x_fake)
        assert x_inv.shape == fake_size
        return {}


class Transpose(AudioTransform):
    invertible = True

    @property
    def scriptable(self):
        return True

    def __repr__(self):
        return "Transpose(dims=%s, contiguous=%s)" % (self.dim, self.contiguous)

    def __init__(self, dims=(-2, -1), contiguous=True):
        super(Transpose, self).__init__()
        self.dims = list(dims)
        self.contiguous = bool(contiguous)

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        data = x.transpose(self.dims[0], self.dims[1])
        if self.contiguous:
            data = data.contiguous()
        return data

    @torch.jit.export
    def invert(self, x: torch.Tensor, inversion_mode: InversionEnumType = None, tolerance: float = 1.e-4):
        return self(x)

    def test_forward(self, x: torch.Tensor, time: torch.Tensor = None):
        fake_size = (2, 128, 512)
        x_fake = self(torch.zeros(fake_size))
        assert x_fake.shape == (2, 512, 128)
        if time is None:
            return x_fake
        else:
            return x_fake, time

    def test_inversion(self, x: torch.Tensor):
        transposed = self.test_forward(x)
        x = self.invert(transposed)
        assert x.shape == (2, 128, 512)
        return {}


class OneHot(AudioTransform):

    @property
    def scriptable(self):
        return True

    @property
    def invertible(self):
        return True

    @property
    def needs_scaling(self):
        return self.n_classes == -1

    def __init__(self, sr=44100, dtype=torch.long, n_classes: int = -1):
        super().__init__(sr)
        self.dtype = dtype
        self.n_classes = n_classes

    def __repr__(self):
        return "OneHot(n_classes=%s)" % self.n_classes

    @torch.jit.export
    def scale_data(self, x: torch.Tensor) -> None:
        self.n_classes = int(x.max()) + 1

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_onehot = torch.nn.functional.one_hot(x, self.n_classes)
        return x_onehot

    @torch.jit.export
    def invert(self, x_onehot: torch.Tensor, inversion_mode: InversionEnumType = None, tolerance: float = 1.e-4) -> torch.Tensor:
        return x_onehot.argmax(-1)

    def test_forward(self, x: torch.Tensor, time: torch.Tensor = None) -> torch.Tensor:
        # simulate ulaw vector
        x = torch.randint(0, 256, (2, 44100))
        self.scale_data(x)
        if time is None:
            x_onehot = self(x)
            return x_onehot
        else:
            x_onehot = self.forward_with_time(x, time)
            return x_onehot

    def test_inversion(self, x: torch.Tensor):
        x_onehot = torch.randint_like(x, 0, 256)
        x = self.invert(x_onehot)
        return {}

    @classmethod
    def test_scripted_transform(cls, transform, invert=True):
        x = torch.randint(0, 256, (2, 44100))
        transform.scale_data(x)
        x_onehot = transform(x)
        if invert:
            x_inv = transform.invert(x)
