import torch
import torch.nn.functional as F
from torchaudio.functional import griffinlim
import math
from .base import AudioTransform, InversionEnumType, _CanTrace, make_fx
from .oadd import OverlapAdd
from typing import Dict, List, Optional, Tuple, Union
from ..utils.misc import *


__all__ = ['STFT', 'RealtimeSTFT', 'StreamingSTFT', 'SPECTRUM_MODES', 'WINDOWS']

MAX_NFFT = 16384
eps = torch.finfo(torch.get_default_dtype()).eps
DB_EPS = 1e-10

SPECTRUM_MODES = ["complex", "magnitude", "power", "db", "phase", "real", "imaginary", "power_density",
                  "instantaneous_phase", "instantaneous_frequency"]
WINDOWS = ["hann", "hamming", "blackman", "bartlett", "rectangular"]
PAD_MODES = ["constant", "reflect", "replicate", "circular"]


def make_window(name: str, n: int) -> torch.Tensor:
    """analysis window of n samples : "hann", "hamming", "blackman", "bartlett" (periodic) or "rectangular"."""
    if name == "hann":
        return torch.hann_window(n)
    if name == "hamming":
        return torch.hamming_window(n)
    if name == "blackman":
        return torch.blackman_window(n)
    if name == "bartlett":
        return torch.bartlett_window(n)
    if name == "rectangular":
        return torch.ones(n)
    raise ValueError("unknown window " + name)


def princarg(phase: torch.Tensor) -> torch.Tensor:
    """wraps phases in [-pi, pi[."""
    return torch.remainder(phase + math.pi, 2 * math.pi) - math.pi


def _parse_modes(mode: Union[str, List[str]]) -> List[str]:
    modes = [mode] if isinstance(mode, str) else list(mode)
    assert len(modes) > 0, "at least one mode must be given"
    for m in modes:
        assert m in SPECTRUM_MODES, "mode must be in %s, got %s" % (SPECTRUM_MODES, m)
    return modes


def realtime_frames(x: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    """frames (..., n_frames, frame_length) of audio (..., T), as given by OverlapAdd (without its buffer)."""
    return x.unfold(-1, frame_length, hop_length)


def overlap_add_frames(frames: torch.Tensor, hop_length: int) -> torch.Tensor:
    """audio (..., T) from synthesis frames (..., n_frames, frame_length), as OverlapAdd.invert (without its buffer)."""
    n_frames, frame_length = frames.size(-2), frames.size(-1)
    length = (n_frames - 1) * hop_length + frame_length
    out = F.fold(frames.reshape(-1, n_frames, frame_length).transpose(-1, -2), output_size=[1, length],
                 kernel_size=[1, frame_length], stride=[1, hop_length])
    return out.reshape(frames.shape[:-2] + (length,)) / (float(frame_length) / hop_length)


class STFT(AudioTransform):
    """Short-time Fourier transform : rfft of windowed frames of win_length samples (n_fft if None), zero-padded at
    their end to n_fft, every hop_length samples. With center, signals are padded with win_length // 2 samples on both
    sides (pad_mode : "constant" for zeros, "reflect", "replicate" or "circular"), such that frame t is centered on
    sample t * hop_length ; with the defaults (win_length = n_fft, center, reflect), it is torch.stft. Inputs shorter
    than a frame are zero-padded to one frame. Outputs are (..., n_frames, n_fft // 2 + 1).

    mode is a mode, or a list of modes stacked on a new axis ((..., n_modes, n_frames, n_fft // 2 + 1)). Outputs are
    complex if "complex" is given (other modes are then cast to complex values), and real otherwise. Modes :
        - "complex" : complex STFT (default).
        - "magnitude", "power", "db" (10 log10 power) : magnitude and power spectra.
        - "phase", "real", "imaginary" : phase (radians, in [-pi, pi]), real and imaginary parts.
        - "power_density" : power spectral density (V^2 / Hz) of each frame, as scipy.signal.spectrogram(
          scaling="density", mode="psd") : power / (sr * sum(window^2)), doubled for one-sided bins.
        - "instantaneous_phase" : phase difference between consecutive frames, wrapped in [-pi, pi] ; 0 for the first
          frame.
        - "instantaneous_frequency" : instantaneous frequency of each bin in Hz (phase vocoder) ; center frequencies of
          bins for the first frame.

    invert gives back audio from outputs : exactly from complex outputs, (real, imaginary) or (magnitude / power / db /
    power_density, phase) ; approximately with instantaneous phases or frequencies (integrated phases) ; with
    inversion_mode (griffin_lim, random, keep_input, sinebank) from magnitudes only. realtime() gives the frame-wise
    RealtimeSTFT, and streaming() a StreamingSTFT transforming chunks of audio.
    """

    @property
    def scriptable(self):
        return True

    @property
    def invertible(self):
        return True

    @property
    def tracable(self):
        return _CanTrace

    @property
    def needs_scaling(self):
        return False

    def __repr__(self):
        repr_str = f"STFT(n_fft={self.n_fft}, hop_length={self.hop_length}, win_length={self.win_length}, " \
                   f"window={self.window_name}, center={self.center}, mode={self.mode}, " \
                   f"inversion_mode = {self.inversion_mode})"
        return repr_str

    def __init__(self,
                 sr: int = 44100,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 dtype: torch.dtype = None,
                 inversion_mode: str = "griffin_lim",
                 window: Optional[str] = "hann",
                 win_length: Optional[int] = None,
                 center: bool = True,
                 pad_mode: str = "reflect",
                 mode: Union[str, List[str]] = "complex"):
        super().__init__(sr=sr)
        dtype = dtype or torch.get_default_dtype()
        self.register_buffer("window", torch.zeros(MAX_NFFT))
        self.register_buffer("inv_window", torch.zeros(MAX_NFFT))
        self.register_buffer("density_scale", torch.zeros(MAX_NFFT // 2 + 1))
        self.register_buffer("eps",  torch.tensor(
            torch.finfo(dtype).eps, dtype=dtype))
        self.register_buffer("phase_buffer", torch.zeros(0))

        # set up window
        window = "rectangular" if window is None else window
        if window not in WINDOWS:
            raise ValueError("Window %s is not known ; available windows : %s" % (window, WINDOWS))
        self.window_name = window
        if pad_mode not in PAD_MODES:
            raise ValueError("pad_mode must be in %s, got %s" % (PAD_MODES, pad_mode))
        self.center = bool(center)
        self.pad_mode = pad_mode
        self.mode = _parse_modes(mode)
        if (n_fft is not None):
            assert hop_length is not None, "n_fft and hop_length must be given together"
        if (hop_length is not None):
            assert n_fft is not None, "n_fft and hop_length must be given together"
        # set up parameters
        self.win_length = int(win_length or n_fft or 0)
        if (n_fft is not None) and (hop_length is not None):
            self.set_params(n_fft, hop_length, win_length)
        if inversion_mode in type(self).get_inversion_modes():
            self.inversion_mode = inversion_mode
        else:
            raise ValueError("Inversion mode %s not known" % inversion_mode)

    @torch.jit.export
    def set_params(self, n_fft: int, hop_length: int, win_length: Optional[int] = None) -> None:
        if win_length is None:
            win_length = n_fft
        assert win_length <= n_fft, "win_length must be <= n_fft"
        assert n_fft <= self.window.size(0), "n_fft must be <= MAX_NFFT"
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window.zero_()
        self.inv_window.zero_()
        self.window[:self.win_length] = self._get_window()
        self.inv_window[:self.win_length] = self._get_dual_window()
        self.density_scale.zero_()
        self.density_scale[:self.n_fft // 2 + 1] = self._get_density_scale()
        self.gamma = self._get_gamma()

    def _get_gamma(self) -> float:
        return 2*math.pi*((-self.n_fft**2/(8*math.log(0.01)))**.5)**2

    def _get_window(self) -> torch.Tensor:
        return make_window(self.window_name, self.win_length)

    def _get_dual_window(self) -> torch.Tensor:
        return self._get_window()

    def _get_density_scale(self) -> torch.Tensor:
        """scaling of power spectra to power spectral densities (V^2 / Hz), as scipy.signal.periodogram(scaling="density")."""
        n_bins = self.n_fft // 2 + 1
        window_power = float(self._get_window().square().sum())
        scale = torch.full((n_bins,), 1. / (self.sr * window_power))
        # one-sided spectra : bins are doubled, except 0 and the Nyquist bin (even n_fft)
        last = n_bins if self.n_fft % 2 == 1 else n_bins - 1
        scale[1:last] *= 2
        return scale

    @property
    def ratio(self):
        return self.hop_length

    @torch.jit.export
    def set_inversion_mode(self, inversion_mode: str) -> None:
        if inversion_mode in self.get_inversion_modes():
            self.inversion_mode = inversion_mode
        else:
            raise AttributeError('inversion mode %s not valid'%inversion_mode)

    # analysis / synthesis

    def _pad(self, x: torch.Tensor, n: int) -> torch.Tensor:
        if self.pad_mode == "constant" or (self.pad_mode == "reflect" and x.size(-1) <= n):
            return F.pad(x, [n, n])
        return F.pad(x.unsqueeze(-2), [n, n], mode=self.pad_mode).squeeze(-2)

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T) -> complex spectrum (B, n_frames, n_fft // 2 + 1)."""
        win_length = self.win_length
        if self.center:
            x = self._pad(x, win_length // 2)
        if x.size(-1) < win_length:
            x = F.pad(x, [0, win_length - x.size(-1)])
        frames = x.unfold(-1, win_length, self.hop_length) * self.window[:win_length].to(x.dtype)
        return torch.fft.rfft(frames, n=self.n_fft)

    def _istft(self, spectrum: torch.Tensor) -> torch.Tensor:
        """complex spectrum (B, n_frames, n_fft // 2 + 1) -> (B, T) : windowed overlap-add, normalized by the sum of
        squared windows (samples covered by vanishing windows, e.g. at the borders without center, are set to 0)."""
        win_length, hop_length = self.win_length, self.hop_length
        window = self.window[:win_length].to(spectrum.real.dtype)
        frames = torch.fft.irfft(spectrum, n=self.n_fft)[..., :win_length] * window
        n_frames = frames.size(-2)
        length = (n_frames - 1) * hop_length + win_length
        out = F.fold(frames.transpose(-1, -2), output_size=[1, length], kernel_size=[1, win_length], stride=[1, hop_length])
        norm = F.fold(window.square().view(1, win_length, 1).expand(1, win_length, n_frames), output_size=[1, length],
                      kernel_size=[1, win_length], stride=[1, hop_length])
        out, norm = out.reshape(-1, length), norm.reshape(1, length)
        out = torch.where(norm > 1e-10, out / norm.clamp(min=1e-10), torch.zeros_like(out))
        if self.center:
            out = out[..., win_length // 2: length - win_length // 2]
        return out

    # modes

    def _bins(self, like: torch.Tensor) -> torch.Tensor:
        return torch.arange(self.n_fft // 2 + 1, dtype=like.dtype, device=like.device)

    def _phase_delta(self, spectrum: torch.Tensor, previous_phase: Optional[torch.Tensor]) -> torch.Tensor:
        """phase differences between frames (..., n_frames, K) ; 0 for the first frame without previous_phase."""
        phase = spectrum.angle()
        if previous_phase is None:
            previous = phase[..., :1, :]
        else:
            previous = previous_phase
        phase = torch.cat([previous, phase], dim=-2)
        return princarg(phase[..., 1:, :] - phase[..., :-1, :])

    def _to_modes(self, spectrum: torch.Tensor, previous_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """complex spectrum (..., n_frames, K) -> outputs of modes (..., [n_modes], n_frames, K)."""
        outputs: List[torch.Tensor] = []
        n_bins = spectrum.size(-1)
        for mode in self.mode:
            if mode == "complex":
                out = spectrum
            elif mode == "magnitude":
                out = spectrum.abs()
            elif mode == "power":
                out = spectrum.abs().square()
            elif mode == "db":
                out = 10 * torch.log10(spectrum.abs().square() + 1e-10)    # DB_EPS
            elif mode == "phase":
                out = spectrum.angle()
            elif mode == "real":
                out = spectrum.real
            elif mode == "imaginary":
                out = spectrum.imag
            elif mode == "power_density":
                out = spectrum.abs().square() * self.density_scale[:n_bins].to(spectrum.real.dtype)
            else:
                delta = self._phase_delta(spectrum, previous_phase)
                if mode == "instantaneous_phase":
                    out = delta
                else:
                    # phase vocoder : phase advance of the bin center, plus the deviation from it
                    expected = 2 * math.pi * self._bins(delta) * self.hop_length / self.n_fft
                    out = (expected + princarg(delta - expected)) * self.sr / (2 * math.pi * self.hop_length)
                    if previous_phase is None:
                        first = (self._bins(delta) * self.sr / self.n_fft).expand_as(out[..., :1, :])
                        out = torch.cat([first, out[..., 1:, :]], dim=-2)
            outputs.append(out)
        if "complex" in self.mode:
            outputs = [o if o.is_complex() else torch.complex(o, torch.zeros_like(o)) for o in outputs]
        if len(outputs) == 1:
            return outputs[0]
        return torch.stack(outputs, dim=-3)

    def _real(self, x: torch.Tensor) -> torch.Tensor:
        return x.real if x.is_complex() else x

    def _from_modes(self, x: torch.Tensor, initial_phase: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """outputs of modes (..., n_modes, n_frames, K) -> (complex spectrum, magnitude, integrated phase) : complex
        spectrum if it can be rebuilt, and magnitude otherwise. Phases of instantaneous modes are integrated from
        initial_phase (0 if None)."""
        magnitude: Optional[torch.Tensor] = None
        phase: Optional[torch.Tensor] = None
        delta: Optional[torch.Tensor] = None
        real: Optional[torch.Tensor] = None
        imaginary: Optional[torch.Tensor] = None
        for i in range(len(self.mode)):
            mode = self.mode[i]
            y = x.select(-3, i)
            if mode == "complex":
                if y.is_complex():
                    return y, None, None
                magnitude = y
            elif mode == "magnitude":
                magnitude = self._real(y)
            elif mode == "power":
                magnitude = self._real(y).clamp(min=0).sqrt()
            elif mode == "db":
                magnitude = (torch.pow(10., self._real(y) / 10) - 1e-10).clamp(min=0).sqrt()    # DB_EPS
            elif mode == "power_density":
                scale = self.density_scale[:y.size(-1)].to(self._real(y).dtype)
                magnitude = (self._real(y) / scale).clamp(min=0).sqrt()
            elif mode == "phase":
                phase = self._real(y)
            elif mode == "real":
                real = self._real(y)
            elif mode == "imaginary":
                imaginary = self._real(y)
            elif mode == "instantaneous_phase":
                delta = self._real(y)
            elif mode == "instantaneous_frequency":
                delta = self._real(y) * 2 * math.pi * self.hop_length / self.sr
        if real is not None and imaginary is not None:
            return torch.complex(real, imaginary), None, None
        if magnitude is None:
            raise ValueError("modes %s cannot be inverted : a magnitude (or real and imaginary parts) is needed" % (self.mode))
        integrated: Optional[torch.Tensor] = None
        if phase is None and delta is not None:
            phase = torch.cumsum(delta, dim=-2)
            if initial_phase is not None:
                phase = phase + initial_phase
            integrated = phase
        if phase is not None:
            return torch.polar(magnitude, phase), None, integrated
        return None, magnitude, None

    # public API

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, batch_shape = reshape_batches(x, -1)
        spectrum = self._stft(x)
        self._replace_phase_buffer(spectrum.angle())
        out = self._to_modes(spectrum, None)
        return out.reshape(batch_shape + out.shape[1:])

    @torch.jit.export
    def forward_with_time(self, x: torch.Tensor, time: torch.Tensor):
        transform = self.forward(x)
        n_chunks = transform.size(-2)
        shifts = torch.arange(n_chunks) * self.hop_length / self.sr
        new_shape = [t for t in transform.shape[:-2]]
        new_shape.append(n_chunks)
        new_strides = [0] * (x.ndim-1)
        new_strides.append(1)
        shifts = shifts.as_strided(new_shape, new_strides)
        new_time = shifts + time.unsqueeze(-1)
        return transform, new_time

    def _n_event_dims(self) -> int:
        return 2 if len(self.mode) == 1 else 3

    @torch.jit.export
    def invert(self, x: torch.Tensor, inversion_mode: InversionEnumType = None) -> torch.Tensor:
        x, batch_shape = reshape_batches(x, -self._n_event_dims())
        if len(self.mode) == 1:
            x = x.unsqueeze(-3)
        spectrum, magnitude, _ = self._from_modes(x, None)
        if spectrum is not None:
            x_inv = self._istft(spectrum)
        else:
            assert magnitude is not None
            x_inv = self.invert_without_phase(magnitude, inversion_mode)
        return x_inv.reshape(batch_shape + x_inv.shape[-1:])

    @staticmethod
    def get_inversion_modes():
        return ['griffin_lim', 'keep_input', 'random', 'sinebank']

    def _replace_phase_buffer(self, phase: torch.Tensor) -> None:
        self.phase_buffer = phase

    def _get_phase_buffer(self, mag: torch.Tensor) -> torch.Tensor:
        if mag.shape[:-2] != self.phase_buffer.shape[:-2]:
            phase_buffer = torch.tensor(0)
            self.phase_buffer = phase_buffer
        else:
            phase_buffer = self.phase_buffer
        return phase_buffer

    def _input_phase(self, mag: torch.Tensor) -> torch.Tensor:
        """phase of the last forward call if it matches mag, and random phase otherwise."""
        if self.phase_buffer.shape == mag.shape:
            return self.phase_buffer.to(mag.dtype)
        return 2 * math.pi * torch.rand_like(mag)

    def realtime(self):
        inversion_mode = self.inversion_mode if self.inversion_mode in RealtimeSTFT.get_inversion_modes() else "random"
        return RealtimeSTFT(sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, inversion_mode=inversion_mode,
                            window=self.window_name, win_length=self.win_length, mode=self.mode)

    def streaming(self):
        """StreamingSTFT with the same parameters, transforming chunks of audio (see StreamingSTFT)."""
        inversion_mode = self.inversion_mode if self.inversion_mode in RealtimeSTFT.get_inversion_modes() else "random"
        return StreamingSTFT(sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, inversion_mode=inversion_mode,
                             window=self.window_name, win_length=self.win_length, mode=self.mode)

    def invert_without_phase(self, x: torch.Tensor, inversion_mode: InversionEnumType = None) -> torch.Tensor:
        if inversion_mode is None:
            inversion_mode = self.inversion_mode
        if (inversion_mode == "keep_input"):
            return self._istft(torch.polar(x, self._input_phase(x)))
        if (inversion_mode == "griffin_lim"):
            return self._griffin_lim(x)
        elif (inversion_mode == "random"):
            return self._istft(torch.polar(x, 2 * math.pi * torch.rand_like(x)))
        elif (inversion_mode == "sinebank"):
            return self.get_sinebank_inversion(x)
        else:
            raise ValueError("inversion mode %s not valid." % inversion_mode)

    def _griffin_lim(self, magnitude: torch.Tensor, n_iter: int = 32, momentum: float = 0.99) -> torch.Tensor:
        """fast Griffin-Lim (Perraudin et al., 2013) with the framing of the transform."""
        angles = torch.polar(torch.ones_like(magnitude), 2 * math.pi * torch.rand_like(magnitude))
        previous = torch.zeros_like(angles)
        for _ in range(n_iter):
            rebuilt = self._stft(self._istft(magnitude * angles))
            angles = rebuilt - (momentum / (1 + momentum)) * previous
            angles = angles / (angles.abs() + 1e-16)
            previous = rebuilt
        return self._istft(magnitude * angles)

    def griffin_lim(self, x: torch.Tensor) -> torch.Tensor:
        n_fft = self.n_fft
        hop_length = self.hop_length
        window = self.inv_window[:n_fft]
        return griffinlim(x.transpose(-2, -1), window, n_fft, hop_length, n_fft, 1.0, 30, 0.99, None, True)

    def get_sinebank_inversion(self, x_fft):
        bpad = (1,) * len(x_fft.shape[:-2])
        fft_frequencies = torch.linspace(0, self.sr/2, int(self.n_fft / 2 + 1)).view(bpad+(-1, 1))
        random_phase = 2 * torch.pi * torch.rand(fft_frequencies.shape[-2], 1)
        x_fft = x_fft / x_fft.abs().max()
        final_length = self.hop_length * x_fft.shape[-2] + self.n_fft
        t = torch.linspace(0, final_length / self.sr, final_length).view(bpad+(1, -1))
        enveloppes = torch.nn.functional.interpolate(x_fft.transpose(-2, -1), final_length, mode="linear") / (2 * torch.pi)
        x = enveloppes * torch.sin(2 * torch.pi * fft_frequencies * t + random_phase)
        x = x.sum(-2)
        x = x / x.max()
        return x

    # TESTS
    def test_inversion(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outs = {}
        x_stft = self.forward(x)
        outs['direct'] = self.invert(x_stft)
        if getattr(self, "mode", ["complex"]) == ["complex"]:
            for inv_type in self.get_inversion_modes():
                outs[inv_type] = self.invert(
                    x_stft.abs(), inversion_mode=inv_type)
        return outs

    @classmethod
    def test_scripted_transform(cls, transform: AudioTransform, invert: bool = True, batch_size = (2, 2)):
        x = torch.zeros(*batch_size, 44100)
        time = torch.zeros(*batch_size)
        x_t = transform.forward(x)
        x_t, time_t = transform.forward_with_time(x, time)
        if invert:
            x_inv = transform.invert(x_t)
            for inv_type in cls.get_inversion_modes():
                x_inv = transform.invert(x_t.abs(), inversion_mode=inv_type)

    # inversion modes that cannot be traced (data-dependent python control flow or state)
    untracable_inversion_modes = ["keep_input"]

    @classmethod
    def _tracing_input(cls, transform: AudioTransform, x: torch.Tensor) -> torch.Tensor:
        """inputs of forward from audio (..., T) : audio for offline transforms."""
        return x

    @classmethod
    def _tracing_output(cls, transform: AudioTransform, y: torch.Tensor) -> torch.Tensor:
        """audio (..., T) from outputs of invert."""
        return y

    @classmethod
    def _before_tracing(cls, transform: AudioTransform, x: torch.Tensor) -> None:
        """called before tracing invert on x (e.g. to set states of streaming transforms with concrete shapes)."""
        pass

    @classmethod
    def test_tracable_transform(cls, transform: AudioTransform, test_files):
        n_samples = transform.n_fft * 16
        test_files = cls._tracing_input(transform, test_files[0])
        x_tracing = torch.randn(test_files.shape[0] + 1, test_files.shape[1], n_samples)
        x_tracing = cls._tracing_input(transform, x_tracing)
        fns = {}
        outs = {}

        fns['forward'] = make_fx(transform, tracing_mode="symbolic", _allow_non_fake_inputs = True)(x_tracing)
        y_python = transform(test_files)
        y_tracing = fns['forward'](x_tracing)
        y_orig = fns['forward'](test_files)

        assert torch.allclose(y_python, y_orig)
        f = make_fx(transform, _fn="invert", tracing_mode="symbolic", _allow_non_fake_inputs = True)(y_tracing)
        fns['invert_from_complex'] = f
        outs['from_complex_traced'] = f(y_python)
        outs['from_complex_original'] = transform.invert(y_python)

        assert torch.allclose(outs['from_complex_traced'], outs['from_complex_original'])
        for inv_type in transform.get_inversion_modes():
            if inv_type in cls.untracable_inversion_modes: continue
            cls._before_tracing(transform, y_tracing.abs())
            f = make_fx(transform, _fn="invert", tracing_mode="symbolic", _allow_non_fake_inputs = True)(y_tracing.abs(), inv_type)
            outs[f'{inv_type}_original'] = transform.invert(y_tracing.abs(), inv_type)
            outs[f'{inv_type}_traced'] = f(y_tracing.abs(), inv_type)
            # assert torch.allclose(outs[f'{inv_type}_original'] , outs[f'{inv_type}_traced'])
            fns['invert_%s'%inv_type] = f

        outs = {k: cls._tracing_output(transform, v) for k, v in outs.items()}
        return fns, outs



class RealtimeSTFT(STFT):
    """frame-wise STFT : transforms frames (..., n_frames, win_length) (e.g. given by OverlapAdd(win_length,
    hop_length)) into (..., [n_modes], n_frames, n_fft // 2 + 1), and inverts them back into frames to overlap-add.
    Frames are the ones of a STFT without center : after the n_fft / hop_length - 1 first frames of a stream (that
    contain the zeros of OverlapAdd buffers), frames are the ones of STFT(center=False). Phases of the last frame are
    kept between calls for instantaneous modes. Synthesis frames are weighted by the dual window, scaled for the gain
    compensation of OverlapAdd, such that OverlapAdd.invert(invert(forward(OverlapAdd(x)))) gives back x (delayed by
    win_length - hop_length samples)."""

    def __init__(self, sr: int = 44100, n_fft: int = 1024, hop_length: int = 256, dtype: torch.dtype = None,
                 inversion_mode: InversionEnumType = "random", window: Optional[str] = "hann", batch_size: int = 2,
                 win_length: Optional[int] = None, mode: Union[str, List[str]] = "complex"):
        super().__init__(sr=sr, n_fft=n_fft, hop_length=hop_length, dtype=dtype, inversion_mode=inversion_mode,
                         window=window, win_length=win_length, center=False, pad_mode="constant", mode=mode)
        self.batch_size = batch_size
        self.register_buffer('random_phase', 2 * torch.pi * torch.rand(int(self.n_fft / 2 + 1)))
        self.register_buffer('time_index', torch.tensor(0.))
        # phases of the last analyzed frame, and of the last synthesized frame (instantaneous modes)
        self.register_buffer('last_phase', torch.zeros(0))
        self.register_buffer('last_inv_phase', torch.zeros(0))

    @property
    def scriptable(self):
        return True

    @property
    def invertible(self):
        return True

    @property
    def needs_scaling(self):
        return False

    def __repr__(self):
        repr_str = f"RealtimeSTFT(n_fft={self.n_fft}, hop_length={self.hop_length}, win_length={self.win_length}, " \
                   f"window={self.window_name}, mode={self.mode}, inversion_mode = {self.inversion_mode})"
        return repr_str

    @staticmethod
    def get_inversion_modes():
        return ['keep_input', 'random', 'sinebank']

    @classmethod
    def _tracing_input(cls, transform: AudioTransform, x: torch.Tensor) -> torch.Tensor:
        """frame-wise transforms take frames (..., n_frames, frame_length), as given by OverlapAdd."""
        return realtime_frames(x, getattr(transform, "win_length", transform.n_fft), transform.hop_length)

    @classmethod
    def _tracing_output(cls, transform: AudioTransform, y: torch.Tensor) -> torch.Tensor:
        return overlap_add_frames(y, transform.hop_length)

    def _get_dual_window(self) -> torch.Tensor:
        """canonical dual window w / sum_k w(n + k hop)^2, scaled by win_length / hop_length (gain compensation of
        OverlapAdd)."""
        window = self._get_window()
        square = window.square()
        norm = torch.zeros(self.win_length)
        offset = 0
        while offset < self.win_length:
            norm[:self.win_length - offset] += square[offset:]
            if offset > 0:
                norm[offset:] += square[:self.win_length - offset]
            offset += self.hop_length
        return window * (float(self.win_length) / self.hop_length) / norm.clamp(min=1e-10)

    @torch.jit.export
    def reset(self) -> None:
        self.time_index = torch.zeros_like(self.time_index)
        self.last_phase = torch.zeros(0, device=self.last_phase.device)
        self.last_inv_phase = torch.zeros(0, device=self.last_inv_phase.device)

    def _uses_instantaneous_modes(self) -> bool:
        return ("instantaneous_phase" in self.mode) or ("instantaneous_frequency" in self.mode)

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        win_length = self.win_length
        spectrum = torch.fft.rfft(x * self.window[:win_length].to(x.dtype), n=self.n_fft)
        self._replace_phase_buffer(spectrum.angle())
        previous_phase: Optional[torch.Tensor] = None
        if self._uses_instantaneous_modes():
            if self.last_phase.dim() == spectrum.dim() and self.last_phase.shape[:-2] == spectrum.shape[:-2]:
                previous_phase = self.last_phase
            self.last_phase = spectrum.angle()[..., -1:, :]
        return self._to_modes(spectrum, previous_phase)

    @torch.jit.export
    def forward_with_time(self, x: torch.Tensor, time: torch.Tensor):
        return self(x), time

    def _synthesis(self, spectrum: torch.Tensor) -> torch.Tensor:
        win_length = self.win_length
        return torch.fft.irfft(spectrum, n=self.n_fft)[..., :win_length] * self.inv_window[:win_length].to(spectrum.real.dtype)

    @torch.jit.export
    def invert(self, x: torch.Tensor, inversion_mode: InversionEnumType = None) -> torch.Tensor:
        if len(self.mode) == 1:
            x = x.unsqueeze(-3)
        initial_phase: Optional[torch.Tensor] = None
        if self.last_inv_phase.dim() == x.dim() - 1 and self.last_inv_phase.shape[:-2] == x.shape[:-3]:
            initial_phase = self.last_inv_phase
        spectrum, magnitude, integrated = self._from_modes(x, initial_phase)
        if integrated is not None:
            self.last_inv_phase = integrated[..., -1:, :]
        if spectrum is not None:
            return self._synthesis(spectrum)
        assert magnitude is not None
        return self.invert_without_phase(magnitude, inversion_mode)

    @torch.jit.export
    def get_batch_size(self, batch_size: int):
        return batch_size

    @torch.jit.export
    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def get_sinebank_inversion(self, x_fft):
        batch_shape = x_fft.shape[:-2]
        bpad = (1,) * len(batch_shape)
        if (batch_shape != self.random_phase.shape[:-2]):
            self.random_phase = 2 * torch.pi * torch.rand(batch_shape + (1, x_fft.size(-1)))
        # construct phase
        t = torch.arange(self.n_fft).unsqueeze(0) + torch.arange(x_fft.size(-2)).unsqueeze(1) * self.hop_length
        t = (t / self.sr + self.time_index).view(bpad + (t.shape[0], 1, t.shape[1]))
        # get frequencies
        fft_frequencies = torch.linspace(0, self.sr/2, int(self.n_fft / 2 + 1)).view(bpad + (1, -1, 1))
        # get sines
        sines = torch.sin(2 * torch.pi * fft_frequencies * t + self.random_phase.view(batch_shape + (1, -1, 1)))
        x = x_fft.unsqueeze(-1) * sines
        x = x.sum(-2) / x_fft.size(-1)
        self.time_index += (x_fft.size(-2) * self.hop_length + self.n_fft) / self.sr
        return x

    def invert_without_phase(self, x: torch.Tensor, inversion_mode: InversionEnumType = None) -> torch.Tensor:
        if inversion_mode is None:
            inversion_mode = self.inversion_mode
        if (inversion_mode == "keep_input"):
            phase = self._input_phase(x)
        elif (inversion_mode == "random"):
            phase = 2 * math.pi * torch.rand_like(x)
        elif (inversion_mode == "sinebank"):
            win_length = self.win_length
            return self.get_sinebank_inversion(x)[..., :win_length] * self.inv_window[:win_length]
        else:
            raise ValueError("inversion mode %s not valid." %
                    self.inversion_mode)
        return self._synthesis(torch.polar(x, phase))

    # TESTS
    def test_forward(self, x: torch.Tensor, time: torch.Tensor = None):
        x = frame(x, self.win_length, self.hop_length, -1)
        transform = []
        for i in range(x.shape[-2]):
            transform.append(self(x[..., i:i+1, :]))
        transform = torch.cat(transform, -2)
        if time is None:
            return transform
        else:
            return transform, None

    def test_inversion(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.reset()
        win_length = self.win_length
        chunk_size = win_length * 4
        hop_length = self.hop_length
        oadd = OverlapAdd(win_length, hop_length)
        outs = {}
        x_framed = x.split(chunk_size, -1)
        # test direct inversion
        outs["direct"] = []
        for n in range(len(x_framed)):
            x_in = oadd(x_framed[n])
            x_t = self(x_in)
            x_i = oadd.invert(self.invert(x_t))
            outs['direct'].append(x_i)
        outs['direct'] = torch.cat(outs['direct'], -1)
        # test spectrogram inversion modes
        if self.mode == ["complex"]:
            for inv_mode in ['sinebank']:#self.get_inversion_modes():
                self.inversion_mode = inv_mode
                oadd = OverlapAdd(win_length, hop_length)
                outs[inv_mode] = []
                for n in range(len(x_framed)):
                    x_in = oadd(x_framed[n])
                    x_t = self(x_in)
                    x_i = oadd.invert(self.invert(x_t.abs(), inversion_mode=inv_mode))
                    outs[inv_mode].append(x_i)
                outs[inv_mode] = torch.cat(outs[inv_mode], -1)
        return outs


    @classmethod
    def test_scripted_transform(cls, transform: AudioTransform, invert: bool = True):
        x = torch.zeros(2, 4, transform.win_length)
        x_t = transform(x)
        if invert:
            x_inv = transform.invert(x_t)
            for inv_type in cls.get_inversion_modes():
                x_inv = transform.invert(x_t.abs(), inversion_mode=inv_type)


class StreamingSTFT(AudioTransform):
    """streaming STFT, for real-time use : forward transforms chunks of audio (..., T) into the spectral frames they
    complete (..., [n_modes], T // hop_length, n_fft // 2 + 1), keeping the last win_length - hop_length samples
    between calls (OverlapAdd), and invert gives back chunks of audio from chunks of frames. Chunks must have a multiple
    of hop_length samples. Frames are the ones of STFT(center=False) on the whole stream, preceded by
    win_length // hop_length - 1 frames that contain the initial zeros of the buffer ; inverted audio is delayed by
    latency = win_length - hop_length samples. Scriptable with torch.jit.script ; call reset() between streams."""

    @property
    def scriptable(self):
        return True

    @property
    def invertible(self):
        return True

    @property
    def needs_scaling(self):
        return False

    def __init__(self, sr: int = 44100, n_fft: int = 1024, hop_length: int = 256, dtype: torch.dtype = None,
                 inversion_mode: InversionEnumType = "random", window: Optional[str] = "hann",
                 win_length: Optional[int] = None, mode: Union[str, List[str]] = "complex"):
        super().__init__(sr=sr)
        self.stft = RealtimeSTFT(sr=sr, n_fft=n_fft, hop_length=hop_length, dtype=dtype, inversion_mode=inversion_mode,
                                 window=window, win_length=win_length, mode=mode)
        assert self.stft.win_length % hop_length == 0, "win_length must be a multiple of hop_length for streaming"
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = self.stft.win_length
        self.latency = self.win_length - hop_length
        self.analysis = OverlapAdd(self.win_length, hop_length)
        self.synthesis = OverlapAdd(self.win_length, hop_length)

    def __repr__(self):
        return "StreamingSTFT(n_fft=%d, hop_length=%d, win_length=%d, window=%s, mode=%s, inversion_mode=%s)" % (
            self.n_fft, self.hop_length, self.win_length, self.stft.window_name, self.stft.mode, self.stft.inversion_mode)

    @staticmethod
    def get_inversion_modes():
        return RealtimeSTFT.get_inversion_modes()

    @torch.jit.export
    def reset(self) -> None:
        self.stft.reset()
        self.analysis.input_buffer = torch.zeros(self.latency, device=self.analysis.input_buffer.device)
        self.synthesis.output_buffer = torch.zeros(self.latency, device=self.synthesis.output_buffer.device)

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stft(self.analysis(x))

    @torch.jit.export
    def forward_with_time(self, x: torch.Tensor, time: torch.Tensor):
        return self.forward(x), time

    @torch.jit.export
    def invert(self, x: torch.Tensor, inversion_mode: InversionEnumType = None) -> torch.Tensor:
        return self.synthesis.invert(self.stft.invert(x, inversion_mode))

    def test_forward(self, x: torch.Tensor, time: torch.Tensor = None):
        self.reset()
        out = torch.cat([self(c) for c in x.split(4 * self.win_length, -1)], -2)
        return out if time is None else (out, None)

    def test_inversion(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.reset()
        out = torch.cat([self.invert(self(c)) for c in x.split(4 * self.win_length, -1)], -1)
        return {"direct": out}

    @classmethod
    def test_scripted_transform(cls, transform: AudioTransform, invert: bool = True):
        x = torch.zeros(2, 4096)
        x_t = transform(x)
        if invert:
            transform.invert(x_t)
            for inv_type in cls.get_inversion_modes():
                transform.invert(x_t.abs(), inversion_mode=inv_type)
