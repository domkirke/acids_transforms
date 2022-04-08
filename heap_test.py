import numpy as np, os, matplotlib.pyplot as plt, math
#import sys; sys.path.insert(0, '/Users/chemla/code/ltfatpy')
import torch, torchaudio
from phase_reconstruction import DGT, import_data
import pghipy

from phase_reconstruction.heap_gradient import OnlinePGHI, hgi

torch.set_num_threads(1)

# files = ["/Users/domkirke/Datasets/acidsInstruments-ordinario/data/Brass/English-Horn/ordinario/EH_nA-4_120-A#3-ff.wav"]
n_fft = 1024
sr = 44100
hop_length = 1024 // 4
normalized = False

if not os.path.isdir("reconstructions"):
    os.makedirs("reconstructions")

x, names = import_data("source_files", sr=44100)

dgt_transform = DGT(n_fft, hop_length)
x_dgt = dgt_transform(x)
x_mag = x_dgt.abs()
x_inv = dgt_transform.invert(x_mag)
"""
for i, n in enumerate(names):
    x_audio = x_inv[i].float()
    torchaudio.save(f"reconstructions/{n}.wav", x_audio, sample_rate=sr)
"""

# real-time test
online_pghi = OnlinePGHI(n_fft, hop_length)
phase = torch.zeros_like(x_mag)
for i in range(x_mag.shape[-2]):
    phase[..., i, :] = online_pghi(x_mag[..., i, :])
x_inv = dgt_transform.invert(x_mag * torch.exp(1j * phase))

fig, ax = plt.subplots(1, 2)
ax[0].imshow(x_dgt.angle()[0, 0], aspect="auto")
ax[1].imshow(phase[0, 0], aspect='auto')
plt.show()



for i, n in enumerate(names):
    x_audio = x_inv[i].float()
    torchaudio.save(f"reconstructions/{n}_rt.wav", x_audio, sample_rate=sr)

# test torchscript
#dgt_transform = ScriptableDGT(n_fft, hop_length)
#dgt_transform(x)
#dgt_transform = torch.jit.script(dgt_transform)

