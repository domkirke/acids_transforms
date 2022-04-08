import numpy as np, os, matplotlib.pyplot as plt
import torch, torchaudio
from phase_reconstruction import griffin_lim, heap_gradient_integration, gaussian_window
import tifresi
torch.set_num_threads(1)

files = ["/Users/chemla/Datasets/toy_additive_mini/data/wav/toy_additive_1000.wav"]
# files = ["/Users/domkirke/Datasets/acidsInstruments-ordinario/data/Brass/English-Horn/ordinario/EH_nA-4_120-A#3-ff.wav"]
n_fft = 1024
window_length = n_fft
hop_length = 1024 // 8
window_fn = gaussian_window
window = window_fn(n_fft)
normalized = False

if not os.path.isdir("reconstructions"):
    os.makedirs("reconstructions")

# fig, ax = plt.subplots(4, 1)
# plot_range = range(0, 1024)
plot_range = range(0, 9000)


x, sr = torchaudio.load(files[0])
# x = x[:, 4096:4096+1024]
# ax[0].plot(x[0][plot_range])

torchaudio.save("reconstructions/original.wav", x, sr)
x_fft = torch.stft(x, n_fft=n_fft, hop_length=hop_length, return_complex=True, 
                                         win_length=window_length, window=window, normalized=normalized)
x_mag = x_fft.abs()

"""
# direct inverse
x_direct = torch.istft(x_fft, n_fft=n_fft, hop_length=hop_length, return_complex=False, win_length=window_length, window=window)
torchaudio.save("reconstructions/direct.wav", x_direct, sr)

# griffin-lim
momentum = 0.99
n_iter = 32
rand_init = False
x_gl_torch = griffin_lim(x_mag, n_fft, hop_length, win_length = window_length, window_fn=window_fn, normalized=False,
                         momentum = momentum, n_iter=n_iter, rand_init=rand_init, backend="torchaudio")
torchaudio.save("reconstructions/griffin_lim_torch.wav", x_gl_torch, sr)
# ax[1].plot(x_gl_torch[0][plot_range])

x_gl_custom = griffin_lim(x_mag, n_fft, hop_length, win_length = window_length, window_fn=window_fn,
                          momentum = momentum, n_iter=n_iter, rand_init=rand_init, backend="custom")
torchaudio.save("reconstructions/griffin_lim_custom.wav", x_gl_custom, sr)
# ax[2].plot(x_gl_custom[0][plot_range])
"""

x_hi = heap_gradient_integration(x_mag[0], n_fft, window_length, hop_length,
                                 order=1, window_fn=torch.hamming_window)
torchaudio.save("reconstructions/griffin_lim_hgi.wav", x_hi.unsqueeze(0), sr)
# ax[3].plot(x_hi[plot_range])

plt.show()





