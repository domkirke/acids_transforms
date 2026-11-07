"""STFT framing options, modes, exact inversion, and streaming (RealtimeSTFT / StreamingSTFT) against the offline STFT."""
import math
import pytest
import torch
from acids_transforms import STFT, RealtimeSTFT, StreamingSTFT, OverlapAdd, RealtimeDGT

SR = 44100


@pytest.fixture
def x():
    # streamed chunks must have a multiple of hop_length samples
    return torch.randn(2, 3, 88 * 256, generator=torch.Generator().manual_seed(0))


def test_defaults_are_torch_stft(x):
    reference = torch.stft(x.reshape(-1, x.shape[-1]), 1024, 256, window=torch.hann_window(1024), return_complex=True)
    reference = reference.transpose(-1, -2).reshape(x.shape[:-1] + reference.shape[-1:-3:-1])
    assert torch.allclose(STFT(n_fft=1024, hop_length=256)(x), reference)


def test_framing():
    x = torch.randn(1, 4000)
    out = STFT(n_fft=2048, win_length=1000, hop_length=250, center=False, mode="complex")(x)
    reference = torch.fft.rfft(x.unfold(-1, 1000, 250) * torch.hann_window(1000), n=2048)
    assert out.shape == reference.shape and torch.allclose(out, reference, atol=1e-4)
    out = STFT(n_fft=1024, win_length=512, hop_length=128, center=True, pad_mode="constant")(x)
    reference = torch.fft.rfft(torch.nn.functional.pad(x, (256, 256)).unfold(-1, 512, 128) * torch.hann_window(512), n=1024)
    assert torch.allclose(out, reference, atol=1e-4)
    # inputs shorter than a frame
    assert STFT(n_fft=1024, hop_length=256, center=False)(torch.randn(1, 100)).shape == (1, 1, 513)


@pytest.mark.parametrize("kwargs", [{}, {"center": False}, {"win_length": 600, "hop_length": 150},
                                    {"win_length": 512, "n_fft": 2048, "hop_length": 128, "pad_mode": "constant"},
                                    {"window": "rectangular", "hop_length": 1024}, {"window": "blackman"}])
def test_inversion(x, kwargs):
    transform = STFT(**{"n_fft": 1024, "hop_length": 256, **kwargs})
    y = transform.invert(transform(x))
    n, w = y.shape[-1], transform.win_length
    # without center, borders covered by vanishing windows are lost
    lo, hi = (0, n) if transform.center else (w, n - w)
    assert torch.allclose(y[..., lo:hi], x[..., lo:hi], atol=1e-4)


@pytest.mark.parametrize("mode", [["magnitude", "phase"], ["real", "imaginary"], ["db", "phase"], ["power", "phase"],
                                  ["power_density", "phase"], ["complex", "magnitude"]])
def test_modes_inversion(x, mode):
    transform = STFT(mode=mode)
    out = transform(x)
    assert out.shape[-3] == len(mode) and out.is_complex() == ("complex" in mode)
    y = transform.invert(out)
    assert torch.allclose(y, x[..., :y.shape[-1]], atol=1e-4)


def test_magnitude_inversion(x):
    transform = STFT(mode="magnitude", inversion_mode="griffin_lim")
    magnitude = transform(x[..., :8192])
    y = transform.invert(magnitude)
    assert y.shape == x[..., :8192].shape
    assert (transform(y) - magnitude).norm() / magnitude.norm() < 0.3
    for inversion_mode in ["random", "keep_input"]:
        assert transform.invert(magnitude, inversion_mode=inversion_mode).shape == y.shape


def test_instantaneous_frequency():
    t = torch.arange(SR // 2) / SR
    sine = torch.sin(2 * math.pi * 1003.7 * t)[None]
    magnitude, frequency, phase = STFT(n_fft=1024, hop_length=256, center=False, mode=["magnitude", "instantaneous_frequency", "instantaneous_phase"])(sine)[0]
    peak = int(magnitude.mean(0).argmax())
    assert torch.allclose(frequency[1:, peak - 1:peak + 2], torch.tensor(1003.7), atol=0.1)
    assert torch.all(phase[0] == 0) and torch.all(phase.abs() <= math.pi + 1e-5)


@pytest.mark.parametrize("kwargs", [{}, {"win_length": 512, "n_fft": 2048, "hop_length": 128},
                                    {"mode": ["magnitude", "instantaneous_frequency", "power_density"]}])
def test_streaming(x, kwargs):
    kwargs = {"n_fft": 1024, "hop_length": 256, **kwargs}
    streaming, offline = StreamingSTFT(**kwargs), STFT(center=False, **kwargs)
    frames = torch.cat([streaming(c) for c in x.split(2048, -1)], -2)
    reference = offline(x)
    # streams start with win_length // hop_length - 1 frames containing the zeros of the buffer
    k = streaming.win_length // streaming.hop_length - 1
    n = reference.shape[-2]
    assert frames.shape[-2] == x.shape[-1] // streaming.hop_length
    # (the first offline frame of instantaneous modes has no previous frame)
    assert torch.allclose(frames[..., k + 1:k + n, :], reference[..., 1:, :], atol=1e-4)
    if "mode" not in kwargs:
        streaming.reset()
        y = torch.cat([streaming.invert(streaming(c)) for c in x.split(2048, -1)], -1)
        latency = streaming.latency
        assert torch.allclose(y[..., latency:], x[..., :-latency], atol=1e-4)


def test_overlap_add_is_identity(x):
    for n_fft, hop_length in [(1024, 256), (1024, 512), (2048, 256)]:
        oadd = OverlapAdd(n_fft, hop_length)
        y = torch.cat([oadd.invert(oadd(c)) for c in x.split(4096, -1)], -1)
        latency = n_fft - hop_length
        assert torch.allclose(y[..., latency:], x[..., :-latency], atol=1e-5)


def test_realtime_dgt_inversion(x):
    dgt, oadd = RealtimeDGT(n_fft=1024, hop_length=256), OverlapAdd(1024, 256)
    y = torch.cat([oadd.invert(dgt.invert(dgt(oadd(c)))) for c in x[0, :1].split(2048, -1)], -1)
    assert torch.allclose(y[..., 768:], x[0, :1, :-768], atol=1e-4)


def test_realtime():
    realtime = STFT(n_fft=1024, hop_length=256, win_length=512, mode=["magnitude", "phase"]).realtime()
    assert isinstance(realtime, RealtimeSTFT) and realtime.win_length == 512 and realtime.mode == ["magnitude", "phase"]
    assert isinstance(STFT().streaming(), StreamingSTFT)


def test_scripted_streaming(x):
    scripted = torch.jit.script(StreamingSTFT(n_fft=1024, hop_length=256, mode=["magnitude", "phase"]))
    python = StreamingSTFT(n_fft=1024, hop_length=256, mode=["magnitude", "phase"])
    a = torch.cat([scripted(c) for c in x.split(2048, -1)], -2)
    b = torch.cat([python(c) for c in x.split(2048, -1)], -2)
    assert torch.allclose(a, b)
    scripted.reset()
    y = torch.cat([scripted.invert(scripted(c)) for c in x.split(2048, -1)], -1)
    assert torch.allclose(y[..., scripted.latency:], x[..., :-scripted.latency], atol=1e-4)
