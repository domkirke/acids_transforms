# acids_transforms

## Description
A bunch of scriptable audio transforms based on the torchaudio backend, embeddable in Python/C++ with torchscript for real-time purposes.
Available transforms:


**transform name** | **invertible** | **scriptable**
 :--- | :---: | :---: 
**raw**
MuLaw | yes | yes
Window | yes | yes
Mono | yes | yes
Stereo | yes | yes
MidSide | yes | yes
OneHot | yes | yes
**spectral**
STFT / RealtimeSTFT  | yes | yes
DGT / RealtimeDGT | yes | yes
MFCC | no | yes
**representations**
Real / Imaginary  | yes | yes
Magnitude / Phase | yes | yes
Instantaneous Frequency | yes | yes
**normalization**
Normalize | yes | yes
**miscalleneous**
OverlapAdd | yes | yes
Squeeze | yes | yes
Unsqueeze | yes | yes
Transpose | yes | yes


## Installation
```
git clone https://github.com/domkirke/acids_transforms.git
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