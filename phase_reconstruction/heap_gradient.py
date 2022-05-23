import torch, torchaudio, math, itertools, random, numpy as np
from .heapq import heappop, heappush
from .utils import *


eps = torch.finfo(torch.get_default_dtype()).eps

class DGT(torch.nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, dtype=None, **kwargs):
        super().__init__()
        self.queue = PriorityQueue()
        dtype = dtype or torch.get_default_dtype()
        self.register_buffer("n_fft", torch.tensor(n_fft))
        self.register_buffer("hop_length", torch.tensor(hop_length))
        self.register_buffer("window", self._get_window())
        self.register_buffer("inv_window", self._get_dual_window())
        self.register_buffer("gamma", self._get_gamma())
        self.register_buffer("eps",  torch.tensor(torch.finfo(dtype).eps, dtype=dtype))

    def _get_gamma(self):
        return torch.tensor(2*torch.pi*((-self.n_fft**2/(8*math.log(0.01)))**.5)**2)

    def _get_window(self) -> torch.FloatTensor:
        lambda_ = (-self.n_fft**2/(8*math.log(0.01)))**.5
        n = torch.arange(0, 2*self.n_fft+1) - (2*self.n_fft) / 2
        w = torch.exp(-n**2 / (2 * (lambda_ * 2)**2 ))
        return w[1:2*self.n_fft+1:2]

    def _get_dual_window(self):
        gsynth = torch.zeros_like(self.window)
        for l in range(int(self.n_fft)):
            denom = 0
            for n in range(-self.n_fft // self.hop_length, self.n_fft // self.hop_length + 1):
                dl = l - n*self.hop_length
                if dl >= 0 and dl < self.n_fft:
                    denom += self.window[dl]**2
            gsynth[l] = self.window[l]/denom
        return gsynth

    @torch.jit.export
    def forward(self, x: torch.Tensor):
        return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True, onesided=True).transpose(-2, -1)

    @torch.jit.export
    def invert(self, x: torch.Tensor, tolerance: float):
        if not torch.is_complex(x):
            phase = self.pghi(x, tolerance)
            x = x * torch.exp(phase * 1j)
        return torch.istft(x.transpose(-2, -1), n_fft=self.n_fft, hop_length=self.hop_length, window=self.inv_window, onesided=True)

    def pghi(self, mag: torch.Tensor, tolerance: float = 1e-6):
        #l = get_lambda(window_fn.__name__) * n_fft**2
        # if mag.ndim > 2:
        #     return torch.stack([self.hgi(m, n_fft, hop_size, tolerance, gamma, order) for m in mag])
        mag = torch.clamp(mag.clone(), self.eps, None)
        # get derivatives
        tgradw, fgradw = self.modgabphasegrad(mag) 
        # prepare integration
        abstol = torch.tensor(self.eps, dtype=mag.dtype)
        #mag = torch.where(mag >= tolerance, mag, torch.full_like(mag, abstol))
        return self.perform_hgi(mag, tgradw, fgradw, abstol, tolerance)

    def perform_hgi(self, X: torch.Tensor, tgradw: torch.Tensor, fgradw: torch.Tensor, abstol: float = 1e-7, tol: float = 1e-6):
        spectrogram = X
        phase = torch.zeros_like(spectrogram)
        M2 = spectrogram.shape[0]
        N = spectrogram.shape[1]
        max_val = spectrogram.max()  # Find maximum value to start integration
        max_pos_f = torch.nonzero(spectrogram == max_val)[0]
        magnitude_heap = [(-max_val, (max_pos_f[0], max_pos_f[1]))] # Numba requires heap to be initialized with content
        spectrogram[max_pos_f[0], max_pos_f[1]] = abstol
        spectrogram = torch.where(spectrogram < max_val * tol, torch.full_like(spectrogram, abstol), spectrogram)
        while max_val > abstol:
            while len(magnitude_heap) > 0: # Integrate around maximum value until reaching silence
                max_val, max_pos = heappop(magnitude_heap)
                col = max_pos[0]
                row = max_pos[1]
                N_pos = col+1, row
                S_pos = col-1, row
                E_pos = col, row+1
                W_pos = col, row-1
                if max_pos[0] < M2-1 and spectrogram[N_pos[0], N_pos[1]] > abstol:
                    phase[N_pos[0], N_pos[1]] = phase[max_pos[0], max_pos[1]] + (fgradw[max_pos[0], max_pos[1]] + fgradw[N_pos[0], N_pos[1]])/2
                    heappush(magnitude_heap, (-spectrogram[N_pos[0], N_pos[1]], N_pos))
                    spectrogram[N_pos[0], N_pos[1]] = abstol
                if max_pos[0] > 0 and spectrogram[S_pos[0], S_pos[1]] > abstol:
                    phase[S_pos[0], S_pos[1]] = phase[max_pos[0], max_pos[1]] - (fgradw[max_pos[0], max_pos[1]] + fgradw[S_pos[0], S_pos[1]])/2
                    heappush(magnitude_heap, (-spectrogram[S_pos[0], S_pos[1]], S_pos))
                    spectrogram[S_pos[0], S_pos[1]] = abstol
                if max_pos[1] < N-1 and spectrogram[E_pos[0], E_pos[1]] > abstol:
                    phase[E_pos[0], E_pos[1]] = phase[max_pos[0], max_pos[1]] + (tgradw[max_pos[0], max_pos[1]] + tgradw[E_pos[0], E_pos[1]])/2
                    heappush(magnitude_heap, (-spectrogram[E_pos[0], E_pos[1]], E_pos))
                    spectrogram[E_pos[0], E_pos[1]] = abstol
                if max_pos[1] > 0 and spectrogram[W_pos[0], W_pos[1]] > abstol:
                    phase[W_pos[0], W_pos[1]] = phase[max_pos[0], max_pos[1]] - (tgradw[max_pos[0], max_pos[1]] + tgradw[W_pos[0], W_pos[1]])/2
                    heappush(magnitude_heap, (-spectrogram[W_pos[0], W_pos[1]], W_pos))
                    spectrogram[W_pos[0], W_pos[1]] = abstol
            max_val = spectrogram.max()
            max_pos_f = torch.nonzero(spectrogram == max_val)[0]
            heappush(magnitude_heap, (-max_val, (max_pos_f[0], max_pos_f[1])))
            spectrogram[max_pos_f[0], max_pos_f[1]] = abstol
        return phase

    def modgabphasegrad(self, mag):
        fmul = self.gamma/(self.hop_length * self.n_fft)
        Y = torch.empty((mag.shape[0]+2,mag.shape[1]+2),dtype=mag.dtype)
        Y[1:-1,1:-1] = torch.log(mag)
        Y[0,:] = Y[1,:]
        Y[-1,:] = Y[-2,:]
        Y[:,0] = Y[:,1]
        Y[:,-1] = Y[:,-2]
        dxdw = (Y[1:-1,2:]-Y[1:-1,:-2])/2
        dxdt = (Y[2:,1:-1]-Y[:-2,1:-1])/2
        fgradw = dxdw/fmul + (2*torch.pi*self.hop_length/self.n_fft)*torch.arange(int(self.n_fft/2)+1).unsqueeze(0)
        tgradw = -fmul*dxdt + torch.pi
        return (tgradw, fgradw)


class RealtimeDGT(DGT):
    def __init__(self, n_fft=1024, hop_length=256, dtype=None, batch_size=2):
        super().__init__(n_fft, hop_length, dtype) 
        self.register_buffer('mag_buffer', torch.zeros(batch_size, 2, n_fft//2+1))
        self.register_buffer("phase_buffer", torch.zeros(batch_size, n_fft//2+1))

    def _get_dual_window(self):
        gsynth = torch.zeros_like(self.window)
        for l in range(int(self.n_fft)):
            denom = 0
            for n in range(-self.n_fft // self.hop_length, self.n_fft // self.hop_length + 1):
                dl = l - n*self.hop_length
                if dl >= 0 and dl < self.n_fft:
                    denom += self.window[dl]**2
            gsynth[l] = self.window[l]/denom
        return gsynth

    @torch.jit.export
    def batch_size(self) -> int:
        return int(self.mag_buffer.size(0))

    @torch.jit.export
    def reset(self) -> None:
        self.mag_buffer.zero_()
        self.phase_buffer.zero_()

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfft(x * self.window.unsqueeze(0))

    @torch.jit.export
    def invert(self, x: torch.Tensor, tolerance: float = 1.e-6) -> torch.Tensor:
        if not torch.is_complex(x):
            phase = self.pghi(x, tolerance)
            x_rec = x * torch.exp(torch.full(phase.shape, 1j) * phase)
            self.update_buffers(x_rec)
            return torch.fft.irfft(x_rec) * self.inv_window.unsqueeze(0)
        else:
            return torch.fft.irfft(x) * self.inv_window.unsqueeze(0)

    def update_buffers(self, x):
        self.mag_buffer = torch.stack([self.mag_buffer[..., 1, :], x.abs()], -2)
        self.phase_buffer = x.angle()

    def pghi(self, mag: torch.Tensor, tolerance: float = 1e-6):
        mag = torch.cat([self.mag_buffer, mag.unsqueeze(-2)], -2)
        # concatenate to buffer
        mag = torch.clamp(mag.clone(), self.eps, None)
        # get derivatives
        tgradw, fgradw = self.modgabphasegrad(mag) 
        # perform integration
        phases = []
        for i in range(self.batch_size()):
            phase = self.perform_hgi(mag[i], self.phase_buffer[i], tgradw[i], fgradw[i], tolerance)
            phases.append(phase)
        phases = torch.stack(phases)
        return phases

    def modgabphasegrad(self, mag):
        """
        taken from
        https://github.com/andimarafioti/tifresi/blob/676db371d5c472a5f3199506bf3863367a2ecde4/tifresi/phase/modGabPhaseGrad.py#L77
        a : length of time shift
        g : window function
        M : number of frequency channels
        """
        #if gamma is None: gamma = 2*torch.pi*((-n_fft**2/(8*torch.log(0.01)))**.5)**2
        fmul = self.gamma/(self.hop_length * self.n_fft)
        Y = torch.empty((mag.shape[0], mag.shape[1], mag.shape[2]+2),dtype=mag.dtype)
        Y[:, :, 1:-1] = torch.log(mag)
        Y[:, :, 0] = Y[:,:,1]
        Y[:, :,-1] = Y[:,:,-2]
        dxdw = (Y[:, :, 2:] - Y[:, :, :-2])/2
        dxdt = (3* Y[..., 2, :] - 4 * Y[..., 1, :] + Y[..., 0, :])/2
        fgradw = dxdw/fmul + (2*torch.pi*self.hop_length/self.n_fft)*torch.arange(int(self.n_fft/2)+1).unsqueeze(0)
        tgradw = -fmul*dxdt + torch.pi
        return (tgradw[:, 1:-1], fgradw[:, 1:, :])

    def perform_hgi(self, spectrogram: torch.Tensor, previous_phase: torch.Tensor, tgradw: torch.Tensor, fgradw: torch.Tensor,  tol: float = 0.000001):
        abstol = torch.clamp(tol * spectrogram.max(), self.eps, None)
        # Random init phase when X < abstol
        new_phase = torch.where(spectrogram[-1] > abstol, torch.zeros_like(spectrogram[-1]), torch.rand_like(spectrogram[-1]))
        # Create I set
        max_val = spectrogram[1].max()
        if max_val <= abstol:
            return new_phase
        # Init heap
        zero_tensor =  torch.zeros(1).long()
        one_tensor = torch.ones(1).long()
        max_val = spectrogram[1].max()
        max_pos_f = torch.nonzero(spectrogram[1] == max_val)
        magnitude_heap = [(-max_val, (one_tensor, max_pos_f[0].unsqueeze(0)))]
        for item in torch.nonzero(spectrogram[0] > abstol):
            heappush(magnitude_heap, (-spectrogram[0, item[0]], (zero_tensor, item[0].long().unsqueeze(0))))
        if len(magnitude_heap) == 0:
            return new_phase
            # item = max_indices[0]
            # heappush(magnitude_heap, (-spectrogram[1, item[0]], (torch.Tensor([1]).int(), item[0].unsqueeze(0))))
        while max_val > abstol:
            while len(magnitude_heap) > 0:
                max_val, max_pos = heappop(magnitude_heap)
                if max_pos[0] == 0:
                    E_pos = (max_pos[0]+1, max_pos[1])
                    if spectrogram[E_pos[0], E_pos[1]] > abstol:
                        new_phase[E_pos[1]] = previous_phase[max_pos[1]] + 0.5 * (fgradw[0, max_pos[1]] + fgradw[1, max_pos[1]])
                        heappush(magnitude_heap, (-spectrogram[1, E_pos[1]], (E_pos[0], E_pos[1])))
                        spectrogram[E_pos[0], E_pos[1]] = abstol
                if max_pos[0] == 1:
                    if max_pos[1] + 1 < spectrogram.shape[1]:
                        N_pos = (max_pos[0], max_pos[1]+1)
                        if spectrogram[N_pos[0], N_pos[1]] > abstol:
                            new_phase[N_pos[1]] = previous_phase[N_pos[1]] + 0.5 * (tgradw[max_pos[1]]+tgradw[N_pos[1]]) 
                            heappush(magnitude_heap, (-spectrogram[1, N_pos[1]], (N_pos[0], N_pos[1])))
                            spectrogram[N_pos[0], N_pos[1]] = abstol
                    if max_pos[1] > 0:
                        S_pos = (max_pos[0], max_pos[1]-1)
                        if spectrogram[S_pos[0], S_pos[1]] > abstol:
                            new_phase[S_pos[1]] = previous_phase[S_pos[1]] - 0.5 * (tgradw[max_pos[1]]+tgradw[S_pos[1]]) 
                            heappush(magnitude_heap, (-spectrogram[1, S_pos[1]], (S_pos[0], S_pos[1])))
                            spectrogram[S_pos[0], S_pos[1]] = abstol
            max_val = spectrogram[1].max()
            max_pos_f = torch.nonzero(spectrogram[1] == max_val)
            heappush(magnitude_heap, (-max_val, (one_tensor, max_pos_f[0].unsqueeze(0))))
            spectrogram[1, max_pos_f[0]] = abstol
        return new_phase
