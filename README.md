# acids_transforms

## Description
A bunch of scriptable audio transforms based on the torchaudio backend, embeddable in Python/C++ with torchscript for real-time purposes.
Available transforms:


Titre colonne 1 (droite) | Titre colonne 1 (centré) | Titre colonne 1 (gauche)
 :--- | :---: | :---: 
**raw**
MuLaw | | 
Window | |
Mono | | 
Stereo
**spectral**
STFT / RealtimeSTFT
DGT / RealtimeDGT
MFCC
**representations**
Real / Imaginary
Magnitude / Phase
Instantaneous Frequency
**normalization**
Normalize

**Raw**
- MuLaw
- Window
- Mono/Stereo

**Spectral**
- STFT / RealtimeSTFT
- DGT / RealtimeDGT (with PGHI)
- Real / Imaginary / Magnitude 
- Phase / Instantaneous Frequency
- MFCC

**Normalization**
- Unipolar ([0;1])
- Bipolar ([-1;1])
- Gaussian (centered)


## Installation
```
git clone 
cd acids_transforms
pip install -r requirements.txt
python3 setup.py install
```

## Usage
Transforms in `acids_transforms` are `nn.Module` with a `forward` function, and an optional `invert` functions if the transform is invertible. Besides, some of them can also be scripted to TorchScript. They can also be combined using the `__add__` operator, allowing to chain several transforms : 


```python
import torch, torchaudio
from acids_transforms.transforms import *

x, sr = torchaudio.load("test/source_files/additive.wav")

transform = Mono() \
    + DGT(sr=sr, n_fft = 1024, hop_length = 256, inversion_mode="pghi") \
    + Magnitude(mel=True, norm="unipolar", contrast="log1p")

print("invertible : ", transform.invertible)
print("scriptable : ", transform.scriptable)
if transform.scriptable:
    transform = torch.jit.script(transform)

# scale normalization
transform.scale_data(x)
# test the transform!
x_transformed = transform(x)
x_inverted = transform.invert(x_transformed)
torchaudio.save("additive_pghi.wav", x_inverted, sample_rate=sr)
```