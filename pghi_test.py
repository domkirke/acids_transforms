from numbers import Real
import numpy as np, os, matplotlib.pyplot as plt, math, tqdm
#import sys; sys.path.insert(0, '/Users/chemla/code/ltfatpy')
import torch, torchaudio
from phase_reconstruction import import_data, DGT, RealtimeDGT
import pghipy, tqdm

# files = ["/Users/domkirke/Datasets/acidsInstruments-ordinario/data/Brass/English-Horn/ordinario/EH_nA-4_120-A#3-ff.wav"]
sr = 44100
n_fft = 1024
hop_factor = 8
USE_SCRIPT = True 
tolerance = 1.e-6


hop_length = n_fft // hop_factor
if not os.path.isdir("reconstructions"):
    os.makedirs("reconstructions")

x, names = import_data("source_files", sr=sr)

# Non Realtime
# dgt_transform = DGT(n_fft, n_fft // hop_factor)
# scripted_dgt = torch.jit.script(dgt_transform)
# for i in tqdm.tqdm(range(x.shape[0]), desc="Exporting non-realtime PGHI", total=x.shape[0]):
#     if USE_SCRIPT:
#         x_dgt = scripted_dgt(x[i][0])
#         x_inv = scripted_dgt.invert(x_dgt.abs(), tolerance)
#     else:
#         x_dgt = dgt_transform(x[i][0])
#         x_inv = dgt_transform.invert(x_dgt.abs(), tolerance)
#     torchaudio.save(f"reconstructions/{names[i]}.wav", x_inv.unsqueeze(0), sample_rate=sr)

# # Real-time
dgt_transform = RealtimeDGT(n_fft, n_fft // hop_factor, batch_size=x.shape[1])
scripted_dgt = torch.jit.script(dgt_transform)

for n in range(x.shape[0]):
    print("Exporting real-time PGHI for file %s.wav..."%names[n])
    n_steps = x.shape[-1] // (n_fft // hop_factor)
    x_inv = torch.zeros(x[n].shape[0], n_steps * hop_length + n_fft + 1)
    x_current = torch.cat([x[n], torch.zeros(*x[n].shape[:-1], x_inv.shape[1] - x[n].shape[-1])], -1)
    for i in tqdm.tqdm(range(n_steps), total=n_steps):
        if USE_SCRIPT:
            x_tmp = x_current[..., i * hop_length : i * hop_length + n_fft].to(torch.float64)
            x_dgt = scripted_dgt(x_tmp)
            x_inv_tmp = scripted_dgt.invert(x_dgt.abs(), 1.e-4)
            x_inv[..., i * hop_length : i * hop_length + n_fft] += x_inv_tmp
        else:
            x_tmp = x_current[..., i * hop_length : i * hop_length + n_fft].to(torch.float64)
            x_dgt = dgt_transform(x_tmp)
            x_inv_tmp = dgt_transform.invert(x_dgt.abs(), 1.e-4)
            x_inv[..., i * hop_length : i * hop_length + n_fft] += x_inv_tmp
    torchaudio.save(f"reconstructions/{names[n]}_rt.wav", x_inv.float(), sample_rate=sr)

