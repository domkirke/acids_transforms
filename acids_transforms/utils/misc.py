from argparse import ArgumentError
from typing import Tuple, List
from . import heapq
import torch
import os
import torchaudio
import torch.nn as nn

eps = torch.finfo(torch.get_default_dtype()).eps


def unwrap(tensor: torch.Tensor):
    """
    unwrap phase for tensors
    :param tensor: phase to unwrap (seq x spec_bin)
    :return: unwrapped phase
    """
    unwrapped = tensor.clone()
    diff = tensor[..., 1:, :] - tensor[..., :-1, :]
    ddmod = (diff + torch.pi) % (2 * torch.pi) - torch.pi
    mask = (ddmod == -torch.pi).bitwise_and(diff > 0)
    ddmod[mask] = torch.pi
    ph_correct = ddmod - diff
    ph_correct[diff.abs() < torch.pi] = 0
    unwrapped[..., 1:, :] = tensor[..., 1:, :] + torch.cumsum(ph_correct, -2)
    return unwrapped


def import_data(path: str, sr=44100):
    if os.path.isfile(path):
        x, sr_file = torchaudio.load(path)
        if sr_file != sr:
            x = torchaudio.transforms.Resample(sr_file, sr)(x)
        return x, os.path.basename(path)
    elif os.path.isdir(path):
        data = []
        names = []
        for f in os.listdir(path):
            try:
                current_x, n = import_data(f"{path}/{f}")
                data.append(current_x)
                names.append(os.path.splitext(os.path.basename(n))[0])
            except:
                pass
        max_size = max([d.shape[1] for d in data])
        stereo = 2 in [d.shape[0] for d in data]
        for i, d in enumerate(data):
            if d.shape[0] > 1:
                d = d if stereo else d[0].unsqueeze(0)
            else:
                d = torch.cat([d, d]) if stereo else d
            if d.shape[1] <= max_size:
                d = torch.cat(
                    [d, torch.zeros(d.shape[0], max_size - d.shape[1], dtype=d.dtype)], 1)
            data[i] = d
        data = torch.stack(data)
        return data, names
    else:
        raise FileNotFoundError(path)

def format_input_data(x: torch.Tensor, dim=-1) -> Tuple[torch.Tensor, torch.Size]:
    batch_size = x.shape[:dim]
    data_size = x.shape[dim:]

def fdiff_forward(x):
    inst_f = torch.cat([x[..., 0, :].unsqueeze(-2),
                        (x[..., 1:, :] - x[..., :-1, :])/2], dim=-2)
    return inst_f


def fdiff_backward(x):
    x = x.flip(-2)
    inst_f = fdiff_forward(x)
    inst_f = inst_f.flip(-2)
    return inst_f

def fdiff_central(x):
    inst_f = torch.cat([x[..., 0, :].unsqueeze(-2), (x[..., 2:, :] -
                                                     x[..., :-2, :])/4, x[..., -1, :].unsqueeze(-2)], dim=-2)
    return inst_f

def fint_forward(x):
    out = x
    out[..., 1:, :] = out[..., 1:, :] * 2
    out = torch.cumsum(out, -2)
    return out


def fint_backward(x):
    out = x.flip(-2)
    out = fint_forward(out)
    out = out.flip(-2)
    return out


def fint_central(x):
    out = torch.zeros_like(x)
    out[..., 0, :] = x[..., 0, :]
    out[..., -1, :] = x[..., -1, :]
    for i in range(2, x.shape[-2], 2):
        out[..., i, :] = out[..., i-2, :] + 4 * x[..., i-1, :]
    for i in range(x.shape[-2]-1, 0, -2):
        out[..., i-2, :] = out[..., i, :] - 4 * x[..., i-1, :]
    return out


def deriv(mag: torch.Tensor, order: int = 2) -> torch.Tensor:
    """
    https://gitlab.lis-lab.fr/dev/ltfatpy/-/blob/master/ltfatpy/fourier/pderiv.py
    """
    assert order in [2, 4, float('inf')], "order must be 2, 4 or inf"
    L = mag.shape[0]
    if order == 2:
        magd = L * (mag.roll(-1) - mag.roll(1)) / 2
    elif order == 4:
        magd = L * (-mag.roll(-2) + 8*mag.roll(-1) -
                    8*mag.roll(1) + mag.roll(2)) / 12
    elif order == float('inf'):
        if L % 2 == 0:
            n = torch.cat([torch.arange(0, L//2+1), torch.arange(-L//2+1, 0)])
        else:
            n = torch.cat([torch.arange(0, (L+1)//2),
                           torch.arange(-(L-1)//2, 0)])
        n = torch.tile(n, (mag.shape[1], 1)).transpose()
        magd = 2 * torch.pi * \
            torch.fft.ifft(1j*torch.fft.fft(mag, dim=0), dim=0)
    return magd


def get_fft_idx(L):
    if L % 2 == 0:
        n = torch.cat([torch.arange(0, L//2+1), torch.arange(-L//2+1, 0)])
    else:
        n = torch.cat([torch.arange(0, (L+1)//2), torch.arange(-(L-1)//2, 0)])
    return n


def pad(tensor: torch.Tensor, target_size: int, dim: int):
    if tensor.size(dim) > target_size:
        return tensor
    tensor_size = list(tensor.shape)
    tensor_size[dim] = target_size - tensor.shape[dim]
    cat_tensor = torch.zeros(
        tensor_size, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, cat_tensor], dim=dim)


def frame(tensor: torch.Tensor, wsize: int, hsize: int, dim: int):
    if dim < 0:
        dim = tensor.ndim + dim
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    n_windows = tensor.shape[dim] // hsize
    tensor = pad(tensor, n_windows * hsize + wsize,  dim)
    shape = list(tensor.shape)
    shape[dim] = n_windows
    shape.insert(dim+1, wsize)
    # shape = shape[:dim] + (n_windows, wsize) + shape[dim+1:]
    strides = [tensor.stride(i) for i in range(tensor.ndim)]
    strides.insert(dim, hsize)
    # strides = strides[:dim] + (hsize,) + strides[dim:]
    # strides = list(strides[dim:], (strides[dim]*hsize) + [hsize * new_stride] + list(strides[dim+1:])
    return torch.as_strided(tensor, shape, strides)

    
def reshape_batches(x: torch.Tensor, dim: int, allow_clone: bool = True):
    batch_size = x.shape[:dim]
    event_size = x.shape[dim:]
    if x.is_contiguous():
        x = x.view(torch.Size([-1]) + event_size)
    else:
        if allow_clone:
            x = x.reshape(torch.Size([-1]) + event_size)
        else:
            raise ValueError("found non contiguous tensor of size : %s"%x.shape)
    return x, batch_size
