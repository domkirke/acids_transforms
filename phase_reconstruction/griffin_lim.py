from numpy import pi
import torch, torchaudio
from . import torch_pi, unwrap



def griffin_lim(mag, n_fft, hop_length, backend="torchaudio", win_length=None, window_fn=torch.hamming_window,
                momentum=0.99, n_iter=32, rand_init=True, normalized=True):
    win_length = win_length or n_fft
    assert backend in ["torchaudio", "custom"]
    if backend == "torchaudio":
        transform = torchaudio.transforms.GriffinLim(n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                                     window_fn=window_fn, power=1, momentum=momentum, n_iter=n_iter,
                                                     rand_init = rand_init)
        raw = transform(mag)
        raw /= raw.abs().max()
    elif backend == "custom":
        phase = torch.rand_like(mag) * 2 * torch_pi - torch_pi if rand_init else torch.zeros_like(mag)
        for i in range(n_iter):
            fft_tmp = mag * torch.exp(phase * 1j)
            raw_tmp = torch.istft(torch.view_as_real(fft_tmp), n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                  window = window_fn(win_length))
            fft_new_tmp = torch.stft(raw_tmp, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                  window = window_fn(win_length))
            new_phase = torch.view_as_complex(fft_new_tmp).angle()
            phase = new_phase + ((phase - new_phase) * momentum / (1+momentum))
        raw = torch.istft(torch.view_as_real(mag * torch.exp(phase * 1j)), n_fft=n_fft,
                                          hop_length=hop_length, win_length=win_length, window = window_fn(win_length))
    raw /= raw.abs().max()
    return raw.clamp(-1, 1)
