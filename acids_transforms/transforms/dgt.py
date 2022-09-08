import torch
import math
from enum import Enum
from .stft import STFT, RealtimeSTFT
from .base import AudioTransform, InversionEnumType
from ..utils.heapq import heappop, heappush
from ..utils.misc import frame, reshape_batches
from typing import Tuple, Dict, Union, List


__all__ = ['DGT', 'RealtimeDGT']

MAX_NFFT = 16384
eps = torch.finfo(torch.get_default_dtype()).eps


class DGT_INVERSION_MODES(Enum):
    KEEP_INPUT = 0
    GRIFFIN_LIM = 1
    PGHI = 2
    RANDOM = 3


class DGT(STFT):

    def __repr__(self):
        repr_str = f"DGT(n_fft={self.n_fft.item()}, hop_length={self.hop_length.item()}, " \
                   f"inversion_mode = {self.inversion_mode})"
        return repr_str

    def __init__(self, sr: int = 44100, n_fft: int = 1024, hop_length: int = 256, dtype: torch.dtype = None, 
                 inversion_mode: InversionEnumType = "pghi", tolerance: float = 1.e-2):
        AudioTransform.__init__(self, sr)
        dtype = dtype or torch.get_default_dtype()
        self.register_buffer("n_fft", torch.zeros(1).long())
        self.register_buffer("hop_length", torch.zeros(1).long())
        self.register_buffer("window", torch.zeros(MAX_NFFT))
        self.register_buffer("inv_window", torch.zeros(MAX_NFFT))
        self.register_buffer("gamma", torch.zeros(1))
        self.register_buffer("eps",  torch.tensor(
            torch.finfo(dtype).eps, dtype=dtype))
        self.register_buffer("phase_buffer", torch.zeros(0))
        self.register_buffer("tolerance", torch.tensor(tolerance))
        if (n_fft is not None):
            assert hop_length is not None, "n_fft and hop_length must be given together"
        if (hop_length is not None):
            assert n_fft is not None, "n_fft and hop_length must be given together"
        if (n_fft is not None) and (hop_length is not None):
            self.set_params(n_fft, hop_length)
        if inversion_mode in type(self).get_inversion_modes():
            self.inversion_mode = inversion_mode
        else:
            raise ValueError("Inversion mode %s not known" % inversion_mode)

    def realtime(self):
        inversion_mode = self.inversion_mode if self.inversion_mode in RealtimeDGT.get_inversion_modes() else "pghi"
        return RealtimeDGT(sr=self.sr, n_fft=self.n_fft.item(), hop_length=self.hop_length.item(), inversion_mode = inversion_mode)

    @property
    def ratio(self):
        return self.hop_length.item()

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, batch_shape = reshape_batches(x, -1)
        window = self.window[:self.n_fft.item()]
        x_dgt = torch.stft(x, n_fft=self.n_fft.item(), hop_length=self.hop_length.item(
        ), window=window, return_complex=True, onesided=True).transpose(-2, -1)
        self._replace_phase_buffer(x_dgt.angle())
        return x_dgt.reshape(batch_shape + x_dgt.shape[-2:])

    @torch.jit.export
    def forward_with_time(self, x: torch.Tensor, time: torch.Tensor): 
        transform = self.forward(x)
        n_chunks = transform.size(-2)
        shifts = torch.arange(n_chunks) * self.hop_length / self.sr
        new_shape = [t for t in transform.shape[:-2]]
        new_shape.append(n_chunks)
        new_strides = [0] * (x.ndim-1)
        new_strides.append(1)
        shifts = shifts.as_strided(new_shape, new_strides)
        new_time = shifts + time.unsqueeze(-1)
        return transform, new_time

    @torch.jit.export
    def invert(self, x: torch.Tensor, inversion_mode: InversionEnumType = None) -> torch.Tensor:
        x, batch_shape = reshape_batches(x, -2)
        if not torch.is_complex(x):
            x_inv = self.invert_without_phase(x, inversion_mode)
        else:
            window = self.inv_window[:self.n_fft.item()]
            x_inv = torch.istft(x.transpose(-2, -1), n_fft=self.n_fft.item(), hop_length=self.hop_length.item(), window=window, onesided=True)
        return x_inv.reshape(batch_shape + x_inv.shape[-1:])            

    @torch.jit.export
    def set_params(self, n_fft: int, hop_length: int) -> None:
        self.n_fft.fill_(n_fft)
        self.hop_length.fill_(hop_length)
        self.window.zero_()
        self.inv_window.zero_()
        self.window[:self.n_fft.item()] = self._get_window()
        self.inv_window[:self.n_fft.item()] = self._get_dual_window()
        self.gamma = self._get_gamma()

    def _get_gamma(self) -> torch.Tensor:
        return 2*torch.pi*((-self.n_fft**2/(8*math.log(0.01)))**.5)**2

    def _get_window(self) -> torch.FloatTensor:
        lambda_ = (-self.n_fft**2/(8*math.log(0.01)))**.5
        n = torch.arange(0, 2*self.n_fft.item()+1) - (2*self.n_fft.item()) / 2
        w = torch.exp(-n**2 / (2 * (lambda_ * 2)**2))
        return w[1:2*self.n_fft.item()+1:2]

    def _get_dual_window(self) -> torch.Tensor:
        gsynth = torch.zeros(self.n_fft.item())
        for l in range(int(self.n_fft.item())):
            denom = 0
            for n in range(int(-self.n_fft.item()//self.hop_length.item()), int(self.n_fft.item()//self.hop_length.item() + 1)):
                dl = l - n*self.hop_length
                if dl >= 0 and dl < self.n_fft:
                    denom += self.window[dl]**2
            gsynth[l] = self.window[l]/denom
        return gsynth

    def invert_without_phase(self, x: torch.Tensor, inversion_mode: InversionEnumType = None) -> torch.Tensor:
        window = self.inv_window[:self.n_fft.item()]
        phase = torch.tensor(0)
        if inversion_mode is None:
            inversion_mode = self.inversion_mode
        if (inversion_mode == "keep_input"):
            phase = self._get_phase_buffer(x)
            if phase.shape[0] == 0:
                phase = torch.pi * 2 * torch.rand_like(x)
        elif (inversion_mode == "griffin_lim"):
            return self.griffin_lim(x)
        elif (inversion_mode == "pghi"):
            if x.ndim == 3:
                phase_array = []
                for n in range(x.shape[0]):
                    phase_array.append(self.pghi(x[n], self.tolerance))
                phase = torch.stack(phase_array)
            else:
                phase = self.pghi(x, self.tolerance)
        elif (inversion_mode == "random"):
            phase = torch.pi * 2 * torch.rand_like(x)
        elif (inversion_mode == "sinebank"):
            x = self.get_sinebank_inversion(x)
            return x
        else:
            raise ValueError("inversion mode %s not valid." %
                             self.inversion_mode)
        x = x * torch.exp(phase * torch.full(phase.size(),
                                             1j, device=phase.device))
        return torch.istft(x.transpose(-2, -1), n_fft=self.n_fft.item(), hop_length=self.hop_length.item(), window=window, onesided=True)

    def pghi(self, mag: torch.Tensor, tolerance: float = 1.e-4) -> torch.Tensor:
        mag = torch.clamp(mag.clone(), self.eps, None)
        # get derivatives
        tgradw, fgradw = self.modgabphasegrad(mag)
        # prepare integration
        abstol = torch.tensor(self.eps, dtype=mag.dtype)
        return self.perform_hgi(mag, tgradw, fgradw, abstol, tolerance)

    @staticmethod
    def get_inversion_modes():
        return ["pghi", "griffin_lim", "random", "keep_input", "sinebank"]
    
    def perform_hgi(self, X: torch.Tensor, tgradw: torch.Tensor, fgradw: torch.Tensor, abstol: float = 1e-7, tol: float = 1.e-2) -> torch.Tensor:
        spectrogram = X
        phase = torch.zeros_like(spectrogram)
        M2 = spectrogram.shape[0]
        N = spectrogram.shape[1]
        max_val = spectrogram.max()  # Find maximum value to start integration
        max_pos_f = torch.nonzero(spectrogram == max_val)[0]
        magnitude_heap = [(-max_val, (max_pos_f[0], max_pos_f[1]))]
        spectrogram[max_pos_f[0], max_pos_f[1]] = abstol
        spectrogram = torch.where(
            spectrogram < max_val * tol, torch.full_like(spectrogram, abstol), spectrogram)
        while max_val > abstol:
            while len(magnitude_heap) > 0:  # Integrate around maximum value until reaching silence
                max_val, max_pos = heappop(magnitude_heap)
                col = max_pos[0]
                row = max_pos[1]
                N_pos = col+1, row
                S_pos = col-1, row
                E_pos = col, row+1
                W_pos = col, row-1
                if max_pos[0] < M2-1 and spectrogram[N_pos[0], N_pos[1]] > abstol:
                    phase[N_pos[0], N_pos[1]] = phase[max_pos[0], max_pos[1]] + \
                        (fgradw[max_pos[0], max_pos[1]] +
                         fgradw[N_pos[0], N_pos[1]])/2
                    heappush(magnitude_heap,
                             (-spectrogram[N_pos[0], N_pos[1]], N_pos))
                    spectrogram[N_pos[0], N_pos[1]] = abstol
                if max_pos[0] > 0 and spectrogram[S_pos[0], S_pos[1]] > abstol:
                    phase[S_pos[0], S_pos[1]] = phase[max_pos[0], max_pos[1]] - \
                        (fgradw[max_pos[0], max_pos[1]] +
                         fgradw[S_pos[0], S_pos[1]])/2
                    heappush(magnitude_heap,
                             (-spectrogram[S_pos[0], S_pos[1]], S_pos))
                    spectrogram[S_pos[0], S_pos[1]] = abstol
                if max_pos[1] < N-1 and spectrogram[E_pos[0], E_pos[1]] > abstol:
                    phase[E_pos[0], E_pos[1]] = phase[max_pos[0], max_pos[1]] + \
                        (tgradw[max_pos[0], max_pos[1]] +
                         tgradw[E_pos[0], E_pos[1]])/2
                    heappush(magnitude_heap,
                             (-spectrogram[E_pos[0], E_pos[1]], E_pos))
                    spectrogram[E_pos[0], E_pos[1]] = abstol
                if max_pos[1] > 0 and spectrogram[W_pos[0], W_pos[1]] > abstol:
                    phase[W_pos[0], W_pos[1]] = phase[max_pos[0], max_pos[1]] - \
                        (tgradw[max_pos[0], max_pos[1]] +
                         tgradw[W_pos[0], W_pos[1]])/2
                    heappush(magnitude_heap,
                             (-spectrogram[W_pos[0], W_pos[1]], W_pos))
                    spectrogram[W_pos[0], W_pos[1]] = abstol
            max_val = spectrogram.max()
            max_pos_f = torch.nonzero(spectrogram == max_val)[0]
            heappush(magnitude_heap, (-max_val, (max_pos_f[0], max_pos_f[1])))
            spectrogram[max_pos_f[0], max_pos_f[1]] = abstol
        return phase

    def modgabphasegrad(self, mag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fmul = self.gamma/(self.hop_length * self.n_fft)
        Y = torch.empty((mag.shape[0]+2, mag.shape[1]+2), dtype=mag.dtype)
        Y[1:-1, 1:-1] = torch.log(mag)
        Y[0, :] = Y[1, :]
        Y[-1, :] = Y[-2, :]
        Y[:, 0] = Y[:, 1]
        Y[:, -1] = Y[:, -2]
        dxdw = (Y[1:-1, 2:]-Y[1:-1, :-2])/2
        dxdt = (Y[2:, 1:-1]-Y[:-2, 1:-1])/2
        # dxdt = (3 * Y[..., 2:, 1:-1] - 4 * Y[..., 1:-1, 1:-1] + Y[..., 0:-2, 1:-1])/2
        fgradw = dxdw/fmul + (2*torch.pi*self.hop_length/self.n_fft) * \
            torch.arange(int(self.n_fft/2)+1).unsqueeze(0)
        tgradw = -fmul*dxdt + torch.pi
        return (tgradw, fgradw)


class RealtimeDGT(DGT):
    def __init__(self, sr: int = 44100, n_fft=1024, hop_length=256, dtype=None, batch_size: Union[int, List[int]] = 2, inversion_mode: InversionEnumType = "pghi"):
        super().__init__(sr=sr, n_fft=n_fft, hop_length=hop_length, dtype=dtype, inversion_mode=inversion_mode)
        if isinstance(batch_size, int):
            self.batch_size = [batch_size]
        else:
            self.batch_size = batch_size

        self.register_buffer(
            'hgi_mag_buffer', torch.zeros(*self.batch_size, 2, n_fft//2+1))
        self.register_buffer("hgi_phase_buffer",
                             torch.zeros(*self.batch_size, n_fft//2+1))
        self.register_buffer('random_phase', 2 * torch.pi * torch.rand(int(self.n_fft.item() / 2 + 1)))
        self.register_buffer('time_index', torch.tensor(0.)) 

    def __repr__(self):
        repr_str = f"RealtimeDGT(n_fft={self.n_fft.item()}, hop_length={self.hop_length.item()}, " \
                   f"inversion_mode = {self.inversion_mode})"
        return repr_str

    @staticmethod
    def get_inversion_modes():
        return ['random', 'pghi', 'keep_input', 'sinebank']

    @torch.jit.export
    def get_batch_size(self) -> List[int]:
        return [int(b) for b in self.batch_size]

    @torch.jit.export
    def set_batch_size(self, batch_size: Union[int, List[int]]):
        self.reset(batch_size)

    @torch.jit.export
    def batch_size(self) -> List[int]:
        return self.hgi_mag_buffer.shape[:-2]

    @torch.jit.export
    def reset(self, batch_size: Union[int, List[int]]) -> None:
        if isinstance(batch_size, int):
            self.batch_size = [batch_size]
        else:
            self.batch_size = torch.Size(batch_size)
        self.hgi_mag_buffer = torch.zeros(torch.Size(self.batch_size) + torch.Size([2, int(self.n_fft)//2+1]))
        self.hgi_phase_buffer = torch.zeros(torch.Size(self.batch_size) + torch.Size([int(self.n_fft)//2+1]))

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        window = self.window[:self.n_fft.item()]
        x_dgt = torch.fft.rfft(x * window)
        self._replace_phase_buffer(x_dgt.angle())
        return x_dgt

    @torch.jit.export
    def forward_with_time(self, x: torch.Tensor, time: torch.Tensor): 
        return self(x), time

    @torch.jit.export
    def invert(self, x: torch.Tensor, inversion_mode: InversionEnumType = None) -> torch.Tensor:
        if not torch.is_complex(x):
            x_rec = self.invert_without_phase(x, inversion_mode)
            return x_rec
        else:
            inv_window = self.inv_window[:self.n_fft.item()]
            return torch.fft.irfft(x) * inv_window

    def invert_without_phase(self, x: torch.Tensor, inversion_mode: InversionEnumType = None) -> torch.Tensor:
        batch_size = x.shape[:-2]
        if batch_size != self.batch_size:
            self.reset(batch_size)
        window = self.inv_window[:self.n_fft.item()]
        phase = torch.tensor(0)
        if inversion_mode is None:
            inversion_mode = self.inversion_mode
        if (inversion_mode == "keep_input"):
            phase = self._get_phase_buffer(x)
            if phase.shape[0] == 0:
                phase = torch.pi * 2 * torch.rand_like(x)
        elif (inversion_mode == "pghi"):
            phase = self.pghi(x, self.tolerance)
        elif (inversion_mode == "random"):
            phase = torch.pi * 2 * torch.rand_like(x)
        elif (inversion_mode == "sinebank"):
            return self.get_sinebank_inversion(x) * window
        else:
            raise ValueError("inversion mode %s not valid." %
                             self.inversion_mode)
        x = x * torch.exp(phase * torch.full(phase.shape,
                                             1j, device=phase.device))
        self.update_buffers(x)
        return torch.fft.irfft(x) * window

    def update_buffers(self, x):
        if x.shape[-2] > 1:
            self.hgi_mag_buffer = x[..., -2:, :].abs()
        else:
            self.hgi_mag_buffer = torch.stack(
                [self.hgi_mag_buffer[..., 1, :], x[..., -1, :].abs()], -2)
        self.hgi_phase_buffer = x[..., -1, :].angle()

    def pghi(self, mag: torch.Tensor, tolerance: float = 1e-6):
        mag, batch_shape = reshape_batches(mag, -2)
        hgi_mag_buffer, _ = reshape_batches(self.hgi_mag_buffer, -2)
        hgi_phase_buffer, _ = reshape_batches(self.hgi_phase_buffer, -1)
        mag = torch.cat([hgi_mag_buffer, mag], -2)
        # concatenate to buffer
        mag = torch.clamp(mag.clone(), self.eps, None)
        # get derivatives
        tgradw, fgradw = self.modgabphasegrad(mag)
        # perform integration
        phases = []
        for i in range(mag.size(0)):
            phase = self.perform_hgi(
                mag[i], hgi_phase_buffer[i], tgradw[i], fgradw[i], tolerance)
            phases.append(phase)
        phases = torch.stack(phases)
        return phases.reshape(batch_shape + phases.shape[1:])

    def get_sinebank_inversion(self, x_fft):
        batch_shape = x_fft.shape[:-2]
        bpad = (1,) * len(batch_shape)
        if (batch_shape != self.random_phase.shape[:-2]):
            self.random_phase = 2 * torch.pi * torch.rand(batch_shape + (1, x_fft.size(-1)))
        # construct phase
        t = torch.arange(self.n_fft.item()).unsqueeze(0) + torch.arange(x_fft.size(-2)).unsqueeze(1) * self.hop_length.item()
        t = (t / self.sr + self.time_index).view(bpad + (t.shape[0], 1, t.shape[1]))
        # get frequencies
        fft_frequencies = torch.linspace(0, self.sr/2, int(self.n_fft.item() / 2 + 1)).view(bpad + (1, -1, 1))
        # get sines
        sines = torch.sin(2 * torch.pi * fft_frequencies * t + self.random_phase.view(batch_shape + (1, -1, 1)))
        x = x_fft.unsqueeze(-1) * sines
        x = x.sum(-2) / x_fft.size(-1)
        self.time_index += (x_fft.size(-2) * self.hop_length.item() + self.n_fft.item()) / self.sr
        return x

    def _get_gamma(self) -> torch.Tensor:
        return (-self.n_fft**2/(8*math.log(0.01)))**.5
        # return 2*torch.pi*((-self.n_fft**2/(8*math.log(0.01)))**.5)**2
        # return - (torch.pi / 4) * (self.n_fft**2 / math.log(self.window[0])

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
        Y = torch.empty((mag.shape[0], mag.shape[1]+2,
                         mag.shape[2]+2), dtype=mag.dtype)
        Y[:, 1:-1, 1:-1] = torch.log(mag)
        Y[:, :, 0] = Y[:, :, 1]
        Y[:, :, -1] = Y[:, :, -2]
        dxdw = (Y[:, 1:-1, 2:] - Y[:, 1:-1, :-2])/2
        dxdt = (3 * Y[..., 2:, 1:-1] - 4 * Y[..., 1:-1, 1:-1] + Y[..., 0:-2, 1:-1])/2
        fgradw = dxdw/fmul + (2*torch.pi*self.hop_length/self.n_fft) * (torch.arange(math.floor(self.n_fft.item()/2)+1))
        tgradw = -fmul * dxdt + torch.pi
        return (tgradw, fgradw)

    def perform_hgi(self, spectrogram: torch.Tensor, previous_phase: torch.Tensor, tgradw: torch.Tensor, fgradw: torch.Tensor,  tol: float = 1.e-7):
        abstol = torch.clamp(tol * spectrogram.max(), self.eps, None)
        # Random init phase when X < abstol
        phase = torch.zeros_like(spectrogram)
        phase[1] = previous_phase
        phase[2:] = torch.where(spectrogram[2:] > abstol, torch.zeros_like(
            spectrogram[2:]), torch.randn_like(spectrogram[2:]))

        # pad gradients for convenient indices
        pad = torch.zeros(2, tgradw.shape[1], device=tgradw.device, dtype=tgradw.dtype)
        tgradw = torch.cat([pad, tgradw], 0)
        fgradw = torch.cat([pad, fgradw], 0)
        spectrogram_history = spectrogram.clone()

        for f in range(2, spectrogram.shape[0]):
            # Create I set
            max_val = spectrogram[f].max()
            if max_val <= abstol:
                continue
                
            # Init heap
            # zero_tensor = torch.zeros(1).long()
            # one_tensor = torch.ones(1).long()
            zero_tensor = torch.full((1,), f-1).long()
            one_tensor = torch.full((1,), f).long()
            max_val = spectrogram[f].max()
            
            max_pos_f = torch.nonzero(spectrogram[f] == max_val)
            magnitude_heap = [(-max_val, (one_tensor, max_pos_f[0].unsqueeze(0)))]
            for item in torch.nonzero(spectrogram_history[f-1] > abstol):
                heappush(
                    magnitude_heap, (-spectrogram_history[f-1, item[0]], (zero_tensor, item[0].unsqueeze(0))))
            if len(magnitude_heap) == 0:
                continue
            while max_val > abstol:
                while len(magnitude_heap) > 0:
                    max_val, max_pos = heappop(magnitude_heap)
                    if max_pos[0] == f-1:
                        E_pos = (max_pos[0]+1, max_pos[1])
                        if spectrogram[E_pos[0], E_pos[1]] > abstol:
                            phase[E_pos[0], E_pos[1]] = phase[max_pos[0], max_pos[1]] + \
                                0.5 * (tgradw[max_pos[0], max_pos[1]] + tgradw[E_pos[0], E_pos[1]])
                            heappush(
                                magnitude_heap, (-spectrogram[E_pos[0], E_pos[1]], (E_pos[0], E_pos[1])))
                            spectrogram[E_pos[0], E_pos[1]] = abstol
                    if max_pos[0] == f:
                        if max_pos[1] + 1 < spectrogram.shape[1]:
                            N_pos = (max_pos[0], max_pos[1]+1)
                            if spectrogram[N_pos[0], N_pos[1]] > abstol:
                                phase[N_pos[0], N_pos[1]] = phase[max_pos[0], max_pos[1]] + \
                                    0.5 * (fgradw[max_pos[0], max_pos[1]]+fgradw[N_pos[0], N_pos[1]])
                                heappush(
                                    magnitude_heap, (-spectrogram[N_pos[0], N_pos[1]], (N_pos[0], N_pos[1])))
                                spectrogram[N_pos[0], N_pos[1]] = abstol
                        if max_pos[1] - 1 > 0:
                            S_pos = (max_pos[0], max_pos[1]-1)
                            if spectrogram[S_pos[0], S_pos[1]] > abstol:
                                phase[S_pos[0], S_pos[1]] = phase[max_pos[0], max_pos[1]] - \
                                    0.5 * (fgradw[max_pos[0], max_pos[1]]+fgradw[S_pos[0], S_pos[1]])
                                heappush(
                                    magnitude_heap, (-spectrogram[S_pos[0], S_pos[1]], (S_pos[0], S_pos[1])))
                                spectrogram[S_pos[0], S_pos[1]] = abstol
                max_val = spectrogram[f].max()
                max_pos_f = torch.nonzero(spectrogram[f] == max_val)
                heappush(magnitude_heap, (-max_val,
                                        (one_tensor, max_pos_f[0].unsqueeze(0))))
                spectrogram[f, max_pos_f[0]] = abstol
        return phase[2:]

    def test_forward(self, x: torch.Tensor, time: torch.Tensor = None):
        x = frame(x, self.n_fft.item(), self.hop_length.item(), -1)
        transform = []
        for i in range(x.shape[-2]):
            transform.append(self(x[..., i, :]))
        transform = torch.stack(transform, -2)
        if time is None:
            return transform
        else:
            return transform, None

    # Tests
    def test_inversion(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        from .oadd import OverlapAdd
        self.reset(x.shape[:-1])
        n_fft = self.n_fft.item()
        chunk_size = n_fft * 4
        hop_length = self.hop_length.item()
        oadd = OverlapAdd(n_fft, hop_length)
        outs = {}
        x_framed = x.split(chunk_size, -1)
        # test direct inversion
        outs["direct"] = []
        for n in range(len(x_framed)):
            x_in = oadd(x_framed[n])
            x_t = self(x_in)
            x_i = oadd.invert(self.invert(x_t))
            outs['direct'].append(x_i)
        outs['direct'] = torch.cat(outs['direct'], -1)
        # test spectrogram inversion modes
        for inv_mode in self.get_inversion_modes():
            self.inversion_mode = inv_mode
            oadd = OverlapAdd(n_fft, hop_length)
            outs[inv_mode] = []
            for n in range(len(x_framed)):
                x_in = oadd(x_framed[n])
                x_t = self(x_in)
                x_i = oadd.invert(self.invert(x_t.abs(), inversion_mode=inv_mode))
                outs[inv_mode].append(x_i)
            outs[inv_mode] = torch.cat(outs[inv_mode], -1)
        return outs


    @classmethod
    def test_scripted_transform(cls, transform: AudioTransform, invert: bool = True):
        x = torch.zeros(2, 1, transform.n_fft.item())
        transform.reset(list(x.shape[:-1]))
        x_t = transform(x)
        if cls.invertible:
            x_inv = transform.invert(x_t)
            for inv_type in cls.get_inversion_modes():
                x_inv = transform.invert(x_t.abs(), inversion_mode=inv_type)
