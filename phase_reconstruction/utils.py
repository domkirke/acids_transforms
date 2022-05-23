from argparse import ArgumentError
from typing import Tuple
from . import heapq
import torch, os, torchaudio, torch.nn as nn
from . import torch_pi

lambda_hash = {'hamming':0.29794, "hann": 0.25645, "blackmann":0.17954}
eps = torch.finfo(torch.get_default_dtype()).eps

def get_lambda(window_name):
    if isinstance(window_name, str):
        for k, v in lambda_hash.items():
            if k in window_name:
                return v
    return 1.

def unwrap(tensor: torch.Tensor):
    """
    unwrap phase for tensors
    :param tensor: phase to unwrap (seq x spec_bin)
    :return: unwrapped phase
    """
    if isinstance(tensor, list):
        return [unwrap(t) for t in tensor]
    if tensor.ndimension() == 2:
        unwrapped = tensor.clone()
        diff = tensor[1:] - tensor[:-1]
        ddmod = (diff + torch_pi)%(2 * torch_pi) - torch_pi
        mask = (ddmod == -torch_pi).bitwise_and(diff > 0)
        ddmod[mask] = torch_pi
        ph_correct = ddmod - diff
        ph_correct[diff.abs() < torch_pi] = 0
        unwrapped[1:] = tensor[1:] + torch.cumsum(ph_correct, 1)
        return unwrapped
    else:
        return torch.stack([unwrap(tensor[i]) for i in range(tensor.size(0))], dim=0)


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
                d = torch.cat([d, torch.zeros(d.shape[0], max_size - d.shape[1], dtype=d.dtype)], 1)
            data[i] = d
        data = torch.stack(data)
        return data, names
    else:
        raise FileNotFoundError(path)

        
def deriv(mag: torch.Tensor, order: int=2) -> torch.Tensor:
    """
    https://gitlab.lis-lab.fr/dev/ltfatpy/-/blob/master/ltfatpy/fourier/pderiv.py
    """
    assert order in [2, 4, float('inf')], "order must be 2, 4 or inf"
    L = mag.shape[0]
    if order == 2:
        magd = L * (mag.roll(-1)  - mag.roll(1)) / 2
    elif order == 4:
        magd = L * (-mag.roll(-2) + 8*mag.roll(-1) - 8*mag.roll(1) + mag.roll(2)) / 12
    elif order == float('inf'):
        if L % 2 == 0:
            n = torch.cat([torch.arange(0, L//2+1), torch.arange(-L//2+1, 0)])
        else:
            n = torch.cat([torch.arange(0, (L+1)//2), torch.arange(-(L-1)//2, 0)]) 
        n = torch.tile(n, (mag.shape[1], 1)).transpose()
        magd = 2* torch.pi * torch.fft.ifft(1j*torch.fft.fft(mag, dim=0), dim=0)
    return magd


class PriorityQueue(nn.Module):
    def __init__(self):
        super().__init__()
        self.clear()

    def clear(self):
        self.heap = []

    def len(self) -> int:
        return len(self.heap)

    def insert(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """
        if self.keys.size(0) == 0:
            self.keys = key.unsqueeze(0)
            self.values = value.unsqueeze(0)
        else:
            idx = (self.keys < key).nonzero()
            if idx.numel() == 0:
                self.keys = torch.cat([self.keys, key.unsqueeze(0)])
                self.values = torch.cat([self.values, value.unsqueeze(0)])
            else:
                idx = idx.min()
                self.keys = torch.cat([self.keys[:idx], key.unsqueeze(0), self.keys[idx:]])
                self.values = torch.cat([self.values[:idx], value.unsqueeze(0), self.values[idx:]])
        return 
        """
        heapq.heappush(self.heap, (float(key), value))

    def pop(self):
        """
        k, v = self.keys[-1], self.values[-1]
        self.keys = self.keys[:-1]
        self.values = self.values[:-1]
        return k, v
        """
        lastelt = self.heap.pop()    # raises appropriate IndexError if heap is empty
        if len(self.heap) > 0:
            returnitem = self.heap[0]
            self.heap[0] = lastelt
            heapq._siftup(self.heap, 0)
            return returnitem
        return lastelt

def get_fft_idx(L):
    if L % 2 == 0:
        n = torch.cat([torch.arange(0, L//2+1), torch.arange(-L//2+1, 0)])
    else:
        n = torch.cat([torch.arange(0, (L+1)//2), torch.arange(-(L-1)//2, 0)]) 
    return n


def gaussian_window(N, hop_length, L):
    t = torch.arange(-N//2, N//2)
    tfr = N * hop_length
    return torch.exp( - torch.pi * tfr * t**2 )

