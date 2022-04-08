from turtle import forward
import torch, torchaudio, math
from .utils import *


class DGT(torch.nn.Module):

    def __init__(self, n_fft=1024, hop_length=256, **kwargs):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", self.get_window())
        self.register_buffer("inv_window", self.get_dual_window())

    def get_window(self):
        lambda_ = (-self.n_fft**2/(8*math.log(0.01)))**.5
        n = torch.arange(0, 2*self.n_fft+1) - (2*self.n_fft) / 2
        w = torch.exp(-n**2 / (2 * (lambda_ * 2)**2 ))
        return w[1:2*self.n_fft+1:2]

    def get_dual_window(self):
        gsynth = torch.zeros_like(self.window)
        for l in range(int(self.n_fft)):
            denom = 0
            for n in range(-self.n_fft // self.hop_length, self.n_fft // self.hop_length + 1):
                dl = l - n*self.hop_length
                if dl >= 0 and dl < self.n_fft:
                    denom += self.window[dl]**2
            gsynth[l] = self.window[l]/denom
        return gsynth

    def forward(self, x):
        batch_shape = x.shape[:-1]
        x = torch.cat([torch.zeros(*batch_shape, self.n_fft//2, device=x.device, dtype=x.dtype),
                    x,
                    torch.zeros(*batch_shape, self.n_fft, device=x.device, dtype=x.dtype)], -1)
        ffts = []
        L = x.shape[-1] - self.n_fft
        for i in range(0, L, self.hop_length):
            current_slice = torch.fft.rfft(self.window*x[..., i:i+self.n_fft])
            ffts.append(current_slice)
        x_fft = torch.stack(ffts, -2)
        return x_fft
        
    def invert(self, x_fft):
        batch_shape = x_fft.shape[:-2]
        if not torch.is_complex(x_fft):
            x_fft = x_fft * torch.exp(1j * hgi(x_fft, self.n_fft, self.hop_length))
        N = x_fft.size(-2)
        x_mx = torch.fft.irfft(x_fft, self.n_fft)
        x = torch.zeros(*batch_shape, N * self.hop_length + self.n_fft)
        for i in range(N):
            x[..., i*self.hop_length:i*self.hop_length+self.n_fft] += self.inv_window * x_mx[..., i, :]
        return x[..., self.n_fft//2:-self.n_fft]

def perform_hgi(mag, tgradw, fgradw, abstol=1.e-10):
    # definitions
    M2 = mag.shape[0]   
    N = mag.shape[1]
    # initialize integration
    phase = torch.zeros_like(mag)
    magnitude_heap = PriorityQueue()
    max_val = mag.max()
    if max_val <= abstol:
        return torch.zeros_like(mag)
    max_pos = torch.nonzero(mag == max_val, as_tuple=False)
    magnitude_heap.insert(-max_val, max_pos)
    mag[max_pos[0, 0], max_pos[0, 1]] = abstol

    while max_val > abstol:
        while len(magnitude_heap) > 0:
            max_val, max_pos = magnitude_heap.pop()
            max_pos = tuple(max_pos.int().tolist())
            col = max_pos[0]
            row = max_pos[1]
            N_pos = col+1, row
            S_pos = col-1, row
            E_pos = col, row+1
            W_pos = col, row-1
            if max_pos[0] < M2-1 and mag[N_pos] > abstol:
                phase[N_pos] = phase[max_pos] + (fgradw[max_pos] + fgradw[N_pos])/2
                magnitude_heap.insert(-mag[N_pos], torch.Tensor([N_pos]))
                mag[N_pos] = abstol
            if max_pos[0] > 0 and mag[S_pos] > abstol:
                phase[S_pos] = phase[max_pos] - (fgradw[max_pos] + fgradw[S_pos])/2
                magnitude_heap.insert(-mag[S_pos], torch.Tensor([S_pos]))
                mag[S_pos] = abstol
            if max_pos[1] < N-1 and mag[E_pos] > abstol:
                phase[E_pos] = phase[max_pos] + (tgradw[max_pos] + tgradw[E_pos])/2
                magnitude_heap.insert(-mag[E_pos], torch.Tensor([E_pos]))
                mag[E_pos] = abstol
            if max_pos[1] > 0 and mag[W_pos] > abstol:
                phase[W_pos] = phase[max_pos] - (tgradw[max_pos] + tgradw[W_pos])/2
                magnitude_heap.insert(-mag[W_pos], torch.Tensor([W_pos]))
                mag[W_pos] = abstol
        max_val = mag.max()
        max_pos = torch.nonzero(mag == max_val, as_tuple=False)
        magnitude_heap.insert(-max_val, max_pos)
        mag[max_pos[0, 0], max_pos[0, 1]] = abstol
    return phase

def hgi(mag, n_fft, hop_size, tolerance = 1.e-7, gamma=None, order=2):
    #l = get_lambda(window_fn.__name__) * n_fft**2
    if mag.ndim > 2:
        return torch.stack([hgi(m, n_fft, hop_size, tolerance, gamma, order) for m in mag])
    mag = mag.clone()
    if gamma is None: gamma = 2*torch.pi*((-n_fft**2/(8*math.log(0.01)))**.5)**2
    # get derivatives
    tgradw, fgradw = modgabphasegrad(mag, n_fft, hop_size, gamma)
    # prepare integration
    abstol = torch.tensor(1.e-10, dtype=mag.dtype)
    mag = torch.where(mag >= tolerance, mag, torch.full_like(mag, abstol))
    return perform_hgi(mag, tgradw, fgradw)
    

class OnlinePGHI(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, tolerance=1.e-7, batch_size=(1, 1)):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.gamma = 2*torch.pi*((-n_fft**2/(8*math.log(0.01)))**.5)**2
        self.register_buffer('mag_buffer', torch.zeros(*batch_size, 2, n_fft//2-1))

    def reset(self, batch_size = None):
        if batch_size is not None:
            self.batch_size = None
        self.mag_buffer = torch.zeros(*batch_size, 2, self.n_fft//2-1)

    def update_buffers(self, x_log):
        self.mag_buffer = torch.stack([x_log[..., 1:-1], self.mag_buffer[..., 0, :]], -2)

    def forward(self, x: torch.Tensor):
        batch_shape = x.shape[:-1]
        assert batch_shape == self.batch_size, "input must be of batch_size (1, 1)"
        x_log = torch.log(x)
        fmul = self.gamma / (2 * self.hop_length * self.n_fft)
        dphase_w = - fmul * (3 * x_log[..., 1:-1] - 4 * self.mag_buffer[..., 0, :] - self.mag_buffer[..., 1, :])
        # dphase_w = torch.cat([torch.zeros(*batch_shape, 1, dtype=x.dtype, device=x.device),
        #                    dphase_w,
        #                    torch.zeros(*batch_shape, 1, dtype=x.dtype, device=x.device)])
        dphase_t = 1 / (4 * fmul) * (x_log[..., 2:] - x_log[..., :-2]) 
        dphase_t = dphase_t + 2 * torch.pi * self.hop_length * torch.arange(1,  self.n_fft//2) / self.n_fft
        self.update_buffers(x_log)
        return perform_hgi(x_log,  dphase_t, dphase_w)
    
