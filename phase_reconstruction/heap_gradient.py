import findiff
from collections import OrderedDict
import torch
import numpy as np
from numpy import pi, arange
central_1 = torch.Tensor([-0.5, 1, 0.5])
from . import torch_pi, lambda_hash
from .utils import get_lambda, gaussian_window


def heap_gradient_integration(mag, n_fft, win_size, hop_size, tolerance = 1.e-7, window_fn=None, order=1):
    l = get_lambda(window_fn.__name__) * n_fft**2
    # definitions
    a = hop_size
    M = mag.shape[0]
    log_mag = mag.log()
    mag_df = torch.zeros_like(mag); mag_dt = torch.zeros_like(mag)
    diff_f = log_mag[:, 2:] - log_mag[:, :-2]; diff_t = log_mag[2:] - log_mag[:-2]

    a = -l / (M*a*2)
    b = (M*a)/(2*l)

    mag_df[:, 1:-1] = a * diff_f
    mag_dt[1:-1] = b * diff_t
    mag_dt += (2 * pi * a * arange(mag.shape[0]) / M)[:, np.newaxis]
    mag_dt[0].zero_(); mag_dt[-1].zero_()

    mesh_f, mesh_t = torch.meshgrid(torch.arange(0, mag.shape[0]), torch.arange(0, mag.shape[1]))
    indices = torch.stack([mesh_f, mesh_t], dim=-1)

    heap_indices = indices[mag >= tolerance]
    heap_mags = mag[heap_indices[..., 0], heap_indices[..., 1]]
    sorted_heap = torch.argsort(heap_mags, descending=True)
    sorted_mags = heap_mags[sorted_heap]
    sorted_indices = heap_indices[sorted_heap]
    I = OrderedDict()
    for i in range(sorted_mags.shape[0]):
        I[tuple(sorted_indices[i].tolist())] = sorted_mags[i]

    # init heap
    heap = []
    phase = torch.zeros_like(mag)
    phase = torch.where(mag < tolerance, phase, 2*torch_pi*torch.rand_like(phase))
    while len(I)>0:
        if len(heap) == 0:
            heap.append(next(I.__iter__()))
            phase[heap[-1]] = 0
            del I[heap[-1]]
            while len(heap) != 0:
                # print(heap)
                ind = heap.pop(0)
                if (ind[0]+1, ind[1]) in I.keys():
                    ind_tmp = (ind[0]+1, ind[1])
                    phase[ind_tmp] = phase[ind] + 0.5 * (mag_df[ind] + mag_df[ind_tmp])
                    heap.append(ind_tmp)
                    del I[ind_tmp]
                if (ind[0] - 1, ind[1]) in I.keys():
                    ind_tmp = (ind[0] - 1, ind[1])
                    phase[ind_tmp] = phase[ind] - 0.5 * (mag_df[ind] + mag_df[ind_tmp])
                    heap.append(ind_tmp)
                    del I[ind_tmp]
                if (ind[0], ind[1]+1) in I.keys():
                    ind_tmp = (ind[0], ind[1]+1)
                    phase[ind_tmp] = phase[ind] + 0.5 * (mag_dt[ind] + mag_dt[ind_tmp])
                    heap.append(ind_tmp)
                    del I[ind_tmp]
                if (ind[0], ind[1]-1) in I.keys():
                    ind_tmp = (ind[0], ind[1]-1)
                    phase[ind_tmp] = phase[ind] - 0.5 * (mag_dt[ind] + mag_dt[ind_tmp])
                    heap.append(ind_tmp)
                    del I[ind_tmp]

    final_fft = mag * torch.exp(1j*phase)
    window = gaussian_window(n_fft, l)
    raw = torch.zeros(mag.shape[1] * hop_size + n_fft)
    for i in range(mag.shape[1]):
        # mirror fft
        spec_temp = final_fft[:, i]
        current_window = torch.fft.irfft(spec_temp)
        current_window *= window
        raw[i*hop_size:i*hop_size+current_window.shape[0]] += current_window

    raw = raw / raw.abs().max()
    return raw

