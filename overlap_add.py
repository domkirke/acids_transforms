import matplotlib.pyplot as plt
import torch, torch.nn as nn
from acids_transforms import RealtimeSTFT, OverlapAdd
n_fft = 2048
hop_length = 512
buffer_size = 4096
rstft = RealtimeSTFT(44100, n_fft, hop_length, inversion_mode="keep_input")

class DumbModel(nn.Module):
    def forward(self, x: torch.Tensor):
        outs = []
        for i in range(x.size(-2)):
            x_fft = rstft(x[..., i, :])
            x_inverted = rstft.invert(x_fft)
            outs.append(x_inverted)
        return torch.stack(outs, -2)

t = torch.linspace(0, 1, 8*n_fft)
x = torch.sin(2 * torch.pi * 2 * t).unsqueeze(0).repeat(2, 1)
x_split = x.split(buffer_size, -1)
model = DumbModel()
oadd = OverlapAdd(n_fft, hop_length)
oadd = torch.jit.script(oadd)
outs = []
for x_tmp in x_split:
    framed_data = oadd(x_tmp)
    model_out = model(framed_data)
    outs.append(oadd.invert(model_out))
out = torch.cat(outs, -1)

plt.plot(x[0])
plt.plot(out[0])
plt.show()


# window = rstft._get_dual_window()
# n_fft = window.size()[0]
# n_windows = 50
# overlap = 13

# x = torch.zeros(int(n_windows * n_fft / overlap + n_fft))
# for i in range(n_windows):
#     x[int(i*(n_fft/overlap)):int(i*(n_fft/overlap)+n_fft)] += window / (overlap / 2)

# plt.plot(x)
# plt.show()

