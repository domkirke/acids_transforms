import torch
from . import torch_pi

lambda_hash = {'hamming':0.29794, "hann": 0.25645, "blackmann":0.17954}


def get_lambda(window_name):
    if isinstance(window_name, str):
        for k, v in lambda_hash.items():
            if k in window_name:
                return v
    return 1.


def gaussian_window(N, l=None):
    t = torch.arange(-N//2, N//2)
    if l is None:
        l = N**2
    return (l / 2)**0.25 * torch.exp(-torch_pi*t.pow(2) / l)


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


