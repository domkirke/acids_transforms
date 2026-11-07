from argparse import ArgumentError
from typing import Tuple, List

import torch.lib
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
    # if tensor.size(dim) > target_size:
    #     return tensor
    tensor_size = list(tensor.shape)
    tensor_size[dim] = target_size - tensor.shape[dim]
    cat_tensor = torch.zeros(
        tensor_size, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, cat_tensor], dim=dim)


@torch.library.custom_op("acids_transforms::frame", mutates_args=())
def frame(x: torch.Tensor, frame_length: int, hop_length: int, dim: int = -1, librosa_order: bool = False) -> torch.Tensor:
    """
    Create overlapping frames (sliding windows) of length frame_length along axis
    with hop size hop_length.

    Args:
        x: torch.Tensor
        frame_length: window length (int > 0)
        hop_length: hop size (int > 0)
        axis: axis along which to frame (default: -1)
        librosa_order: if True, return shape [..., frame_length, n_frames] to match librosa.util.frame;
                    if False, return shape [..., n_frames, frame_length] (PyTorch unfold default order).

    Returns:
        A strided view of x with overlapping frames.
        Shape:
        - if librosa_order=True:  [..., frame_length, n_frames]
        - else:                   [..., n_frames, frame_length]

    Notes:
        - This is a view; use .contiguous() or .clone() if you need a copy.
        - Raises if `frame_length > length along axis`.
    """
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive integers")

    ndim = x.dim()
    dim = dim if dim >= 0 else (ndim + dim)
    if not (0 <= dim < ndim):
        raise ValueError(f"axis out of range for tensor with dim={ndim}: got {dim}")

    x_m = x.clone().movedim(dim, -1)  # bring target axis to the end
    n = x_m.shape[-1]
    if n < frame_length:
        raise ValueError(f"Input length along axis ({n}) is smaller than frame_length ({frame_length})")

    # Unfold replaces the framed axis by n_frames and appends a new trailing dimension of size frame_length
    y = x_m.unfold(-1, frame_length, hop_length)  # shape: [..., n_frames, frame_length]

    if librosa_order:
        # Move the new trailing frame_length dim before the n_frames dim
        y = y.movedim(-1, -2)  # [..., frame_length, n_frames]

    return y    

@frame.register_fake
def _(x: torch.Tensor, frame_length: int, hop_length: int, dim: int = -1, librosa_order: bool = True) -> torch.Tensor:
    if dim < 0: axis += x.ndim
    shape = list(x.shape[:axis]) + [(x.shape[axis] - frame_length) // frame_length, frame_length] + list(x.shape[axis+1:])
    return torch.empty(shape).to(x)


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
