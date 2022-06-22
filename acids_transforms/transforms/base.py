import torch.nn as nn
import torch
from typing import Union


class NotInvertibleError(Exception):
    pass


InversionEnumType = Union[str, None]


class AudioTransform(nn.Module):
    invertible = True
    scriptable = False
    needs_scaling = False

    def __init__(self, sr=44100):
        super(AudioTransform, self).__init__()
        self.sr = sr

    def __repr__(self):
        return "AudioTransform()"

    def __add__(self, transform):
        if isinstance(transform, ComposeAudioTransform):
            return ComposeAudioTransform(transforms=[self] + transform.transforms)
        elif isinstance(transform, AudioTransform):
            return ComposeAudioTransform(transforms=[self, transform])
        else:
            raise TypeError(
                'AudioTransform cannot be added to type: %s' % type(transform))

    @torch.jit.export
    def scale_data(self, x: torch.Tensor) -> None:
        pass

    @torch.jit.export
    def forward(self, x):
        return x

    def get_inversion_modes(self):
        return None

    @torch.jit.export
    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @torch.jit.export
    def forward_with_time(self, x: torch.Tensor, time: torch.Tensor):
        return self.forward(x), time

    def realtime(self):
        return self

    @property
    def ratio(self):
        return 1

    def test_forward(self, x: torch.Tensor, time: torch.Tensor = None):
        if time is None:
            return self.forward(x)
        else:
            return self.forward_with_time(x, time)

    def test_inversion(self, x: torch.Tensor):
        if not self.invertible:
            raise NotImplementedError
        x_transformed = self.forward(x)
        x_inv = self.invert(x_transformed)
        return {'inverted': x_inv}

    @classmethod
    def test_scripted_transform(cls, transform, batch_size=(2,2), invert=True):
        x = torch.zeros(*batch_size, 44100)
        time = torch.zeros(*batch_size)
        x_t = transform.forward(x)
        x_t, time_t = transform.forward_with_time(x, time)
        if invert:
            x_inv = transform.invert(x_t)


class ComposeAudioTransform(AudioTransform):

    @property
    def invertible(self):
        for t in self.transforms:
            if not t.invertible:
                return False
        return True

    @property
    def needs_scaling(self):
        for t in self.transforms:
            if t.needs_scaling:
                return True
        return False

    @property
    def scriptable(self):
        for t in self.transforms:
            if not t.scriptable:
                return False
        return True

    def __getitem__(self, item):
        return self.transforms[item]

    def __init__(self, transforms=[], sr=44100):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def __repr__(self) -> str:
        return "ComposeAudioTransform(%s)" % [t.__repr__()+"\n" for t in self.transforms]

    def __add__(self, itm):
        if not isinstance(itm, AudioTransform):
            raise TypeError(
                "ComposeAudioTransform can only be added to other AudioTransforms")
        if isinstance(itm, ComposeAudioTransform):
            return ComposeAudioTransform(self.transforms + itm.transforms)
        else:
            return ComposeAudioTransform(self.transforms + [itm])

    def __radd__(self, other):
        if not isinstance(other, AudioTransform):
            raise TypeError(
                "ComposeAudioTransform can only be added to other AudioTransforms")
        if isinstance(other, ComposeAudioTransform):
            return ComposeAudioTransform(other.transforms + self.transforms)
        else:
            return ComposeAudioTransform([other] + self.transforms)

    def realtime(self):
        return ComposeAudioTransform(transforms=[t.realtime() for t in self.transforms], sr=self.sr)

    @property
    def ratio(self):
        ratio = 1
        for t in self.transforms:
            ratio = ratio * t.ratio
        return ratio

    @torch.jit.export
    def scale_data(self, x):
        for t in self.transforms:
            t.scale_data(x)
            x = t(x)

    @torch.jit.export
    def forward(self, x: torch.Tensor):
        for t in self.transforms:
            x = t(x)
        return x

    @torch.jit.export
    def forward_with_time(self, x: torch.Tensor, time: torch.Tensor):
        for t in self.transforms:
            x, time = t.forward_with_time(x, time)
        return x, time

    @torch.jit.export
    def invert(self, x, inversion_mode: InversionEnumType = None, tolerance: float = 1.e-4):
        for t in self.transforms[::-1]:
            x = t.invert(x, inversion_mode=inversion_mode, tolerance=tolerance)
        return x

    def get_inversion_modes(self, idx):
        return type(self.transforms[idx]).get_inversion_modes()

    def test_inversion(self, x: torch.Tensor):
        if not self.invertible:
            raise NotImplementedError
        x_transformed = x
        for t in self.transforms:
            x_transformed = t.invert(x_transformed)
        x_inv = x_transformed
        for t in reversed(self.transforms):
            x_inv = t.invert(x_inv)
        return x_inv


def apply_transform_to_list(transform, data, time=None, **kwargs):
    if time is None:
        outs = [transform(data[i], **kwargs) for i in range(len(data))]
        return outs
    else:
        outs = [transform(data[i], time=time[i], **kwargs)
                for i in range(len(data))]
        return [o[0] for o in outs], [o[1] for o in outs]


def apply_invert_transform_to_list(transform, data, time=None, **kwargs):
    if time is None:
        outs = [transform.invert(data[i], **kwargs) for i in range(len(data))]
        return outs
    else:
        outs = [transform.invert(data[i], time=time[i], **kwargs)
                for i in range(len(data))]
        return [o[0] for o in outs], [o[1] for o in outs]
