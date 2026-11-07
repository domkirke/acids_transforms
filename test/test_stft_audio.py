"""Audio examples of STFT analysis / resynthesis, to check by ear : every offline and streaming path is run on the
sounds of test/source_files, and written to test/reconstructions/stft/<sound>/<example>.wav (with original.wav).
Streamed outputs are realigned (latency removed), such that every file can be compared with the original. Each example
is also checked : SNR with the original for exact reconstructions, spectral convergence of the magnitude for
reconstructions without phase. test/reconstructions/stft/report.md lists files and metrics.

    pytest test/test_stft_audio.py
"""
import math
from pathlib import Path
import pytest
import torch
import torchaudio
from acids_transforms import STFT, StreamingSTFT
from acids_transforms.utils.misc import import_data

SR = 44100
N_FFT, HOP = 1024, 256
CHUNK = 1024                                  # streamed chunks (multiple of HOP)
OUT_DIR = Path("test/reconstructions/stft")

# name : (description, function audio (B, C, T) -> reconstruction (B, C, T'), check, threshold)
#   check "snr" : SNR (dB) with the original must be above threshold
#   check "sc"  : spectral convergence |  |S(y)| - |S(x)|  | / | |S(x)| | must be below threshold
EXAMPLES = {}


def example(name, description, check, threshold):
    def decorator(fn):
        EXAMPLES[name] = (description, fn, check, threshold)
        return fn
    return decorator


# offline STFT

def offline(x, inversion_mode=None, invert_from=None, **kwargs):
    transform = STFT(sr=SR, **{"n_fft": N_FFT, "hop_length": HOP, **kwargs})
    y = transform(x)
    if invert_from is not None:
        y = invert_from(y)
    return transform.invert(y, inversion_mode)


@example("offline_complex", "STFT -> complex -> iSTFT", "snr", 60)
def _(x): return offline(x)

@example("offline_magnitude_phase", "modes (magnitude, phase) -> iSTFT", "snr", 60)
def _(x): return offline(x, mode=["magnitude", "phase"])

@example("offline_db_phase", "modes (db, phase) -> iSTFT", "snr", 60)
def _(x): return offline(x, mode=["db", "phase"])

@example("offline_real_imaginary", "modes (real, imaginary) -> iSTFT", "snr", 60)
def _(x): return offline(x, mode=["real", "imaginary"])

@example("offline_power_density_phase", "modes (power_density, phase) -> iSTFT", "snr", 60)
def _(x): return offline(x, mode=["power_density", "phase"])

@example("offline_win512_nfft2048", "win_length 512 zero-padded to n_fft 2048, hop 128, zero-padded center", "snr", 60)
def _(x): return offline(x, win_length=512, n_fft=2048, hop_length=128, pad_mode="constant")

@example("offline_no_center", "center=False (borders, covered by vanishing windows, are lost)", "snr", 40)
def _(x): return offline(x, center=False)

@example("offline_blackman", "blackman window", "snr", 60)
def _(x): return offline(x, window="blackman")

@example("offline_magnitude_instantaneous_frequency", "modes (magnitude, instantaneous frequency) : integrated phase (approximate)", "sc", 0.6)
def _(x): return offline(x, mode=["magnitude", "instantaneous_frequency"])

@example("offline_griffin_lim", "magnitude only -> griffin-lim (32 iterations)", "sc", 0.35)
def _(x): return offline(x, mode="magnitude", inversion_mode="griffin_lim")

@example("offline_random_phase", "magnitude only -> random phase", "sc", 1.)
def _(x): return offline(x, mode="magnitude", inversion_mode="random")

@example("offline_keep_input", "magnitude only -> phase of the last forward (exact)", "snr", 60)
def _(x): return offline(x, mode="magnitude", inversion_mode="keep_input")

@example("offline_sinebank", "magnitude only -> sine bank (approximate)", "sc", 1.5)
def _(x): return offline(x, mode="magnitude", inversion_mode="sinebank")


# streaming STFT : chunks of CHUNK samples, realigned by the latency

def streaming(x, inversion_mode=None, script=False, **kwargs):
    transform = StreamingSTFT(sr=SR, **{"n_fft": N_FFT, "hop_length": HOP, **kwargs})
    if script:
        transform = torch.jit.script(transform)
    transform.reset()
    # chunks are padded with latency zeros, such that the end of the signal goes out
    x = torch.nn.functional.pad(x, (0, transform.latency + (-x.shape[-1] - transform.latency) % CHUNK))
    y = torch.cat([transform.invert(transform(c), inversion_mode) for c in x.split(CHUNK, -1)], -1)
    return y[..., transform.latency:]


@example("streaming_complex", "streaming STFT -> complex -> streaming iSTFT", "snr", 60)
def _(x): return streaming(x)

@example("streaming_complex_scripted", "torch.jit.script(StreamingSTFT), complex", "snr", 60)
def _(x): return streaming(x, script=True)

@example("streaming_magnitude_phase_win512_nfft2048", "streaming modes (magnitude, phase), win_length 512, n_fft 2048, hop 128", "snr", 60)
def _(x): return streaming(x, mode=["magnitude", "phase"], win_length=512, n_fft=2048, hop_length=128)

@example("streaming_magnitude_instantaneous_frequency", "streaming modes (magnitude, instantaneous frequency) : phase integrated across chunks (approximate)", "sc", 0.6)
def _(x): return streaming(x, mode=["magnitude", "instantaneous_frequency"])

@example("streaming_random_phase", "streaming magnitude only -> random phase", "sc", 1.)
def _(x): return streaming(x, mode="magnitude", inversion_mode="random")

@example("streaming_keep_input", "streaming magnitude only -> phase of the last forward (exact)", "snr", 60)
def _(x): return streaming(x, mode="magnitude", inversion_mode="keep_input")

@example("streaming_sinebank", "streaming magnitude only -> sine bank (approximate)", "sc", 1.5)
def _(x): return streaming(x, mode="magnitude", inversion_mode="sinebank")


# metrics

def snr(x, y):
    n = min(x.shape[-1], y.shape[-1])
    x, y = x[..., :n], y[..., :n]
    return float(10 * torch.log10(x.square().sum() / (x - y).square().sum().clamp(min=1e-20)))


def spectral_convergence(x, y):
    n = min(x.shape[-1], y.shape[-1])
    transform = STFT(sr=SR, n_fft=2048, hop_length=512, mode="magnitude")
    sx, sy = transform(x[..., :n]), transform(y[..., :n])
    return float((sy - sx).norm() / sx.norm())


@pytest.fixture(scope="module")
def sounds():
    x, names = import_data("test/source_files", sr=SR)
    return x, names


@pytest.fixture(scope="module")
def report():
    lines = {}
    yield lines
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "report.md", "w") as f:
        f.write("# STFT audio examples\n\nn_fft %d, hop_length %d (unless given) ; streamed chunks of %d samples.\n\n" % (N_FFT, HOP, CHUNK))
        f.write("| example | description | check | %s |\n|---|---|---|%s\n" % (" | ".join(lines.get("_names", [])), "---|" * len(lines.get("_names", []))))
        for name in EXAMPLES:
            if name in lines:
                f.write(lines[name] + "\n")


def save(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    # approximate reconstructions may exceed full scale : they are normalized rather than clipped
    peak = float(x.abs().max())
    torchaudio.save(str(path), (x / peak * 0.99 if peak > 1 else x).float(), sample_rate=SR)


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_audio_example(sounds, report, name):
    x, names = sounds
    description, fn, check, threshold = EXAMPLES[name]
    torch.manual_seed(0)
    y = fn(x)
    assert y.shape[:-1] == x.shape[:-1] and torch.isfinite(y).all()
    scores = []
    for i, sound in enumerate(names):
        save(OUT_DIR / sound / "original.wav", x[i])
        save(OUT_DIR / sound / f"{name}.wav", y[i])
        scores.append(snr(x[i], y[i]) if check == "snr" else spectral_convergence(x[i], y[i]))
    report["_names"] = names
    unit = "SNR (dB) >= %g" % threshold if check == "snr" else "spectral convergence <= %g" % threshold
    report[name] = "| `%s.wav` | %s | %s | %s |" % (name, description, unit, " | ".join("%.3g" % s for s in scores))
    for sound, score in zip(names, scores):
        if check == "snr":
            assert score >= threshold, "%s on %s : SNR %.1f dB < %g dB" % (name, sound, score, threshold)
        else:
            assert score <= threshold, "%s on %s : spectral convergence %.3f > %g" % (name, sound, score, threshold)
