"""Exports nn~ (https://github.com/acids-ircam/nn_tilde) models that forward then invert incoming audio, to check
streaming transforms in Max/MSP or PureData.

Each model has a single "forward" method (audio -> transform -> invert -> audio). If a transform can be inverted in
several ways, the "inversion_mode" attribute selects it from nn~ (e.g. the message "set inversion_mode random") :
"direct" inverts the full output of the transform (exact), the other modes invert its magnitude only.

Models are written in test/nn_tilde/<name>.ts, with <name>_<inversion_mode>.wav (the test sounds streamed through the
exported model, for listening). To only export the models :
    python test/test_nn_tilde.py [output_dir]
"""
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
import torchaudio

try:
    import nn_tilde
except ImportError:
    pytest.skip("nn_tilde is not installed", allow_module_level=True)

from acids_transforms.transforms import AudioTransform, MidSide, MuLaw, OverlapAdd, RealtimeDGT, StreamingSTFT

OUT_DIR = Path(__file__).parent / "nn_tilde"
SR = 44100
# nn~ buffer sizes (multiples of every hop length below)
BUFFER_SIZES = [512, 2048, 4096]


class StreamingFrames(AudioTransform):
    """streams a frame-wise transform (e.g. RealtimeDGT) between two OverlapAdd : chunks of audio -> frames -> audio,
    delayed by n_fft - hop_length samples."""

    def __init__(self, transform: AudioTransform, n_fft: int, hop_length: int):
        super().__init__(sr=transform.sr)
        self.transform = transform
        self.analysis = OverlapAdd(n_fft, hop_length)
        self.synthesis = OverlapAdd(n_fft, hop_length)
        self.latency = n_fft - hop_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(self.analysis(x))

    def invert(self, x: torch.Tensor, inversion_mode: Optional[str] = None) -> torch.Tensor:
        return self.synthesis.invert(self.transform.invert(x, inversion_mode=inversion_mode))


class NNTildeInversion(nn_tilde.Module):
    """nn~ model that transforms then inverts incoming audio, in its "forward" method. inversion_mode is registered as
    an nn~ attribute if several modes are available ("direct", then the given modes of the transform)."""

    def __init__(self, transform: nn.Module, in_channels: int = 1, inversion_modes: Optional[List[str]] = None,
                 latency: int = 0, test_buffer_size: int = 4096):
        super().__init__()
        self.transform = transform
        self.latency = latency
        self.inversion_modes: List[str] = ["direct"] + list(inversion_modes or [])
        self.inversion_mode: Tuple[str] = (self.inversion_modes[0],)
        if len(self.inversion_modes) > 1:
            self.register_attribute("inversion_mode", self.inversion_modes[0])
        self.register_method("forward", in_channels=in_channels, in_ratio=1, out_channels=in_channels, out_ratio=1,
                             input_labels=[f"(signal) input {i}" for i in range(in_channels)],
                             output_labels=[f"(signal) inverted {i}" for i in range(in_channels)],
                             test_buffer_size=test_buffer_size)

    @torch.jit.export
    def get_inversion_mode(self) -> str:
        return self.inversion_mode[0]

    @torch.jit.export
    def set_inversion_mode(self, inversion_mode: str) -> int:
        if inversion_mode not in self.inversion_modes:
            return -1
        self.inversion_mode = (inversion_mode,)
        return 0

    @torch.jit.export
    def get_latency(self) -> int:
        return self.latency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.transform(x)
        inversion_mode = self.inversion_mode[0]
        if inversion_mode == "direct":
            out = self.transform.invert(y)
        else:
            out = self.transform.invert(y.abs(), inversion_mode=inversion_mode)
        return out.float()


def _streaming_stft(n_fft: int, hop_length: int, **kwargs) -> NNTildeInversion:
    transform = StreamingSTFT(sr=SR, n_fft=n_fft, hop_length=hop_length, **kwargs)
    # inversion modes need a single complex (or magnitude) output : with several modes (e.g. magnitude + phase), the
    # phase would be kept by .abs(), and the inversion mode ignored
    inversion_modes = StreamingSTFT.get_inversion_modes() if len(transform.stft.mode) == 1 else None
    return NNTildeInversion(transform, inversion_modes=inversion_modes, latency=transform.latency)


def _streaming_dgt(n_fft: int, hop_length: int) -> NNTildeInversion:
    transform = StreamingFrames(RealtimeDGT(sr=SR, n_fft=n_fft, hop_length=hop_length), n_fft, hop_length)
    return NNTildeInversion(transform, inversion_modes=RealtimeDGT.get_inversion_modes(), latency=transform.latency)


# name -> (model constructor, whether "direct" inversion is exact)
MODELS: Dict[str, Tuple[Callable[[], NNTildeInversion], bool]] = {
    "stft_1024_256": (lambda: _streaming_stft(1024, 256), True),
    "stft_2048_512": (lambda: _streaming_stft(2048, 512), True),
    "stft_2048_256_win1024": (lambda: _streaming_stft(2048, 256, win_length=1024), True),
    "stft_1024_256_magphase": (lambda: _streaming_stft(1024, 256, mode=["magnitude", "phase"]), True),
    "dgt_1024_256": (lambda: _streaming_dgt(1024, 256), True),
    "mulaw": (lambda: NNTildeInversion(MuLaw(channels=256)), False),
    "midside": (lambda: NNTildeInversion(MidSide(), in_channels=2), True),
}


def export(name: str, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.ts"
    MODELS[name][0]().export_to_ts(str(path))
    return path


def stream(model, x: torch.Tensor, buffer_size: int) -> torch.Tensor:
    """calls model as nn~ does, on successive buffers of x (batch, channels, time)."""
    n = x.size(-1) - x.size(-1) % buffer_size
    with torch.no_grad():
        return torch.cat([model(chunk) for chunk in x[..., :n].split(buffer_size, -1)], -1)


@pytest.fixture
def test_audio(test_files):
    x, names = test_files
    return x[..., :4 * SR] if x.size(-1) > 4 * SR else x, names


def _input(model, x: torch.Tensor) -> torch.Tensor:
    n_channels = int(model.forward_params[0])
    return x[:, :n_channels]


@pytest.mark.parametrize("name", list(MODELS))
def test_nn_tilde_export(name, tmp_path):
    path = export(name, tmp_path)
    model = torch.jit.load(str(path))
    assert model.get_methods() == ["forward"]
    n_modes = len(MODELS[name][0]().inversion_modes)
    attributes = model.get_attributes()
    assert ("inversion_mode" in attributes) == (n_modes > 1)
    if n_modes > 1:
        assert model.get_inversion_mode() == "direct"
        assert model.set_inversion_mode("not_a_mode") == -1
        assert model.get_inversion_mode() == "direct"


@pytest.mark.parametrize("buffer_size", BUFFER_SIZES)
@pytest.mark.parametrize("name", list(MODELS))
def test_nn_tilde_streaming(name, buffer_size, test_audio):
    """the exported model streams the test sounds by nn~ buffers, with every inversion mode (written for listening),
    and gives back its input (delayed by its latency) with direct inversion."""
    x, names = test_audio
    path = export(name)
    model = torch.jit.load(str(path))
    x = _input(model, x)
    latency = model.get_latency()
    outs = {}
    for inversion_mode in MODELS[name][0]().inversion_modes:
        model = torch.jit.load(str(path))
        if inversion_mode != "direct":
            assert model.set_inversion_mode(inversion_mode) == 0
        assert model.get_inversion_mode() == inversion_mode
        out = stream(model, x, buffer_size)
        assert out.shape == x[..., :out.size(-1)].shape
        assert out.dtype == torch.float
        assert torch.isfinite(out).all(), f"{name} ({inversion_mode}) : non finite output"
        outs[inversion_mode] = out
        if inversion_mode == "direct" and MODELS[name][1]:
            target = x[..., :out.size(-1) - latency]
            error = (out[..., latency:] - target).abs().max()
            assert error < 1e-3, f"{name} : direct inversion error {error} (buffer size {buffer_size})"
        if buffer_size == BUFFER_SIZES[-1]:
            for i in range(out.size(0)):
                torchaudio.save(str(OUT_DIR / f"{name}_{inversion_mode}_{names[i]}.wav"), out[i], sample_rate=SR)
    # the attribute must change the inversion (random phases differ from the phases of the input)
    if "random" in outs and "keep_input" in outs:
        assert not torch.allclose(outs["random"], outs["keep_input"]), f"{name} : inversion_mode is ignored"


@pytest.mark.parametrize("name", list(MODELS))
def test_nn_tilde_changing_batch(name, test_audio):
    """mc.nn~ may change the number of voices (the batch size) while streaming."""
    x, _ = test_audio
    model = torch.jit.load(str(export(name)))
    x = _input(model, x)
    for batch_size in [1, 3, 2]:
        out = stream(model, x[:1].expand(batch_size, -1, -1)[..., :8192], 2048)
        assert out.shape == (batch_size, x.size(1), 8192)


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR
    for name in MODELS:
        print("exported", export(name, out_dir))
