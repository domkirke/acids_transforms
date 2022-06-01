from email.mime import audio
import torch
import torchaudio
import os
import pytest
from acids_transforms import transforms, AudioTransform


def get_audio_transforms():
    audio_transforms = list(filter(lambda x: (type(getattr(transforms, x)) == type) and (
        not x in ['AudioTransform', 'ComposeAudioTransform']), dir(transforms)))
    audio_transforms = [getattr(transforms, x) for x in audio_transforms]
    audio_transforms = list(
        filter(lambda x: issubclass(x, AudioTransform), audio_transforms))
    return audio_transforms


def get_scriptable_transforms():
    audio_transforms = get_audio_transforms()
    audio_transforms = list(filter(lambda x: x().scriptable, audio_transforms))
    return audio_transforms


def get_invertible_transforms():
    audio_transforms = get_audio_transforms()
    audio_transforms = list(filter(lambda x: x().invertible, audio_transforms))
    return audio_transforms


@pytest.mark.parametrize("transform", get_audio_transforms())
def test_forward(test_files, transform: AudioTransform):
    transform = transform()
    raw, name = test_files
    for i in range(raw.size(0)):
        time = torch.zeros(1)
        y = transform.test_forward(raw[i])
        y, time = transform.test_forward(raw[i], time)


@pytest.mark.parametrize("transform", get_invertible_transforms())
def test_inversion(test_files, transform: AudioTransform):
    if not os.path.isdir("test/reconstructions"):
        os.makedirs("test/reconstructions")
    raw, names = test_files
    transform_name = transform.__name__.split(".")[-1]
    transform = transform()
    for i in range(raw.size(0)):
        x_inv = transform.test_inversion(raw[i])
        for k, v in x_inv.items():
            if v.ndim == 1:
                v = v.unsqueeze(0)
            torchaudio.save(
                f"test/reconstructions/{names[i]}_{transform_name}_{k}.wav", v, sample_rate=44100)
    return True


@pytest.mark.parametrize("transform_type", get_scriptable_transforms())
def test_scriptable(transform_type):
    transform = transform_type()
    scripted_transform = torch.jit.script(transform)
    transform_type.test_scripted_transform(
        scripted_transform, invert=transform.invertible)
    return True


combinations = {
    "stft+magnitude": transforms.STFT() + transforms.Magnitude(),
    # "scaled+dgt": 10.0 * transforms.DGT(),
    "stereo+mulaw": transforms.Stereo() + transforms.MuLaw(),
    "stft+polar": transforms.STFT() + transforms.Polar()
}


@pytest.mark.parametrize("transform", combinations)
def test_combinations(test_files, transform):
    if not os.path.isdir("test/reconstructions"):
        os.makedirs("test/reconstructions")
    raw, names = test_files
    transform_name = transform
    transform = combinations[transform]
    if transform.scriptable:
        transform = torch.jit.script(transform)
    for i in range(raw.size(0)):
        if transform.needs_scaling:
            transform.scale_data(raw[i])
        x_t = transform.forward(raw[i])
        if transform.invertible:
            x_inv = transform.invert(x_t)
            if x_inv.ndim == 1:
                x_inv = x_inv.unsqueeze(0)
            torchaudio.save(
                f"test/reconstructions/{names[i]}_{transform_name}.wav", x_inv, sample_rate=44100)
    return True
