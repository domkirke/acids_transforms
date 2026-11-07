# acids_transforms

## Description

A bunch of scriptable audio transforms based on the torchaudio backend, embeddable in Python/C++ with TorchScript for real-time purposes.
Available transforms:

**transform name** | **invertible** | **scriptable**
:--- | :---: | :---:
**raw** | |
MuLaw | yes | yes
Window | yes | yes
Mono | yes | yes
Stereo | yes | yes
MidSide | yes | yes
OneHot | yes | yes
**spectral** | |
STFT / RealtimeSTFT / StreamingSTFT | yes | yes
DGT / RealtimeDGT | yes | yes
MFCC | no | yes
**representations** | |
Real / Imaginary | yes | yes
Magnitude / Phase | yes | yes
Instantaneous Frequency (IF) | yes | yes
Polar / PolarIF / Cartesian | yes | yes
**normalization** | |
Normalize | yes | yes
**miscellaneous** | |
OverlapAdd | yes | yes
Squeeze | no | yes
Unsqueeze | yes | yes
Transpose | yes | yes

## Installation

```bash
git clone https://github.com/domkirke/acids_transforms.git
cd acids_transforms
pip install .
```

For development, `pip install -e ".[test]"` installs the package in editable mode, with pytest.

## Usage

Transforms in `acids_transforms` are `nn.Module` with a `forward` function, and an optional `invert` function if the transform is invertible. Besides, some of them can also be scripted to TorchScript. They can also be combined using the `__add__` operator, allowing to chain several transforms :

```python
import torch, torchaudio
from acids_transforms.transforms import *

x, sr = torchaudio.load("test/source_files/additive.wav")

transform = Mono() \
    + DGT(sr=sr, n_fft = 1024, hop_length = 256, inversion_mode="pghi") \
    + Magnitude(mel=True, mode="unipolar", contrast="log1p")

print("invertible : ", transform.invertible)
print("scriptable : ", transform.scriptable)
if transform.scriptable:
    transform = torch.jit.script(transform)

# scale normalization
transform.scale_data(x)
# test the transform!
x_transformed = transform(x)
x_inverted = transform.invert(x_transformed)
torchaudio.save("additive_pghi.wav", x_inverted, sample_rate=sr)
```

## STFT

`STFT` transforms audio `(..., T)` into spectra `(..., n_frames, n_fft // 2 + 1)`. Frames of `win_length` samples (`n_fft` by default) are windowed, zero-padded at their end to `n_fft`, every `hop_length` samples. With `center` (default), signals are padded with `win_length // 2` samples on both sides (`pad_mode` : `"reflect"` by default, `"constant"` for zeros, `"replicate"` or `"circular"`), such that frame `t` is centered on sample `t * hop_length` : with the default parameters, `STFT` gives `torch.stft`. Windows are `"hann"` (default), `"hamming"`, `"blackman"`, `"bartlett"` or `"rectangular"` (or `None`).

`mode` gives the output of the transform, or several outputs stacked on a new axis `(..., n_modes, n_frames, n_fft // 2 + 1)` with a list of modes :

- `"complex"` (default) : complex spectrum ;
- `"magnitude"`, `"power"`, `"db"` (`10 log10` of the power) ;
- `"phase"` (in `[-π, π]`), `"real"`, `"imaginary"` ;
- `"power_density"` : power spectral density (V²/Hz), as `scipy.signal.spectrogram(scaling="density", mode="psd")` ;
- `"instantaneous_phase"` : phase difference between consecutive frames, wrapped in `[-π, π]` ;
- `"instantaneous_frequency"` : frequency (Hz) of each bin from the phase difference between frames (phase vocoder).

Outputs are complex if `"complex"` is in modes (other modes are then cast to complex values), and real otherwise.

`invert` gives back audio :

- exactly, from `"complex"`, (`"real"`, `"imaginary"`), or a magnitude (`"magnitude"`, `"power"`, `"db"`, `"power_density"`) with `"phase"` ;
- approximately, from a magnitude with `"instantaneous_phase"` or `"instantaneous_frequency"` (integrated phases) ;
- from a magnitude only, with `inversion_mode` : `"griffin_lim"` (default, fast Griffin-Lim with the framing of the transform), `"random"` (random phase), `"keep_input"` (phase of the last `forward` call) or `"sinebank"`.

```python
import torch
from acids_transforms import STFT

x = torch.randn(2, 44100)
stft = STFT(n_fft=2048, win_length=1024, hop_length=256, center=False, mode=["magnitude", "phase"])
y = stft(x)                                             # (2, 2, 169, 1025)
x_rec = stft.invert(y)                                  # exact (but on borders, without center)

magnitude = STFT(n_fft=1024, hop_length=256, mode="magnitude", inversion_mode="griffin_lim")
x_gl = magnitude.invert(magnitude(x))                   # griffin-lim
x_random = magnitude.invert(magnitude(x), inversion_mode="random")

scripted = torch.jit.script(stft)                       # forward and invert are scriptable
```

### Real-time

`StreamingSTFT` computes the STFT of a stream, chunk by chunk, for real-time use (e.g. in a TorchScript model) :

- `forward` transforms chunks of audio `(..., T)`, with `T` a multiple of `hop_length`, into the `T // hop_length` frames they complete ; the last `win_length - hop_length` samples are kept between calls. Frames are the ones of `STFT(center=False)` on the whole stream, preceded by `win_length // hop_length - 1` frames that contain the initial zeros of the stream ;
- `invert` gives back chunks of audio from chunks of frames, delayed by `latency = win_length - hop_length` samples. Inversion is exact from complex outputs or magnitudes and phases ; without phase, `inversion_mode` is `"random"` (default), `"keep_input"` or `"sinebank"` ;
- `reset()` clears buffers between streams.

`STFT(...).streaming()` gives the `StreamingSTFT` of a `STFT`, with the same parameters.

```python
import torch
from acids_transforms import STFT

stream = torch.jit.script(STFT(n_fft=1024, hop_length=256, mode=["magnitude", "phase"]).streaming())
x = torch.randn(1, 44032)
out = []
for chunk in x.split(512, -1):                          # chunks of 2 * hop_length samples
    frames = stream(chunk)                              # (1, 2, 2, 513) : (modes, frames, bins)
    out.append(stream.invert(frames))                   # 512 samples, delayed by stream.latency (768) samples
out = torch.cat(out, -1)
assert torch.allclose(out[..., stream.latency:], x[..., :-stream.latency], atol=1e-4)
```

`StreamingSTFT` combines `OverlapAdd`, that frames the stream (`forward`) and overlap-adds synthesis frames (`invert`), and `RealtimeSTFT`, that transforms frames `(..., n_frames, win_length)` into spectra and back (synthesis frames are weighted by the dual window of the analysis window). `RealtimeDGT` can be used with `OverlapAdd` the same way.

## Tests

```bash
pytest test
```

`test/test_stft_audio.py` writes audio examples of every offline and streaming STFT path (exact inversions, Griffin-Lim, random phase, sine bank, integrated instantaneous frequencies...) on the sounds of `test/source_files` into `test/reconstructions/stft/<sound>/`, aligned with `original.wav` for listening ; `test/reconstructions/stft/report.md` lists them with their SNR or spectral convergence.

`test/test_nn_tilde.py` (needs [`nn_tilde`](https://github.com/acids-ircam/nn_tilde)) exports [nn~](https://github.com/acids-ircam/nn_tilde) models into `test/nn_tilde/<name>.ts`, that forward then invert incoming audio (`StreamingSTFT`, `RealtimeDGT` between `OverlapAdd`, `MuLaw`, `MidSide`), to check streaming in Max/MSP or PureData : load them with `[nn~ <name> forward]`. When several inversion modes are available, the `inversion_mode` attribute selects one (`direct`, exact, or `random`, `keep_input`, `sinebank`, `pghi`... from the magnitude only). Tests stream the test sounds through the exported models by nn~ buffers (written next to the models), check exact reconstruction with `direct` and batch size changes (`mc.nn~`). `python test/test_nn_tilde.py [out_dir]` only exports the models.

## Warnings

- on real-time PGHI : while pghi is usable with RealtimeDGT, there are some known issues with a stream use. This will come very soon!
- PGHI inversion (`DGT`, `RealtimeDGT`) cannot be traced with `torch.fx` (`make_fx`) : it relies on data-dependent control flow.
- this library is still experimental and changes very often. a first "sure" release will come soon!
