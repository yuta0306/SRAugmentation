# SRAugmentation
Augmentation for Single-Image-Super-Resolution

## Requirements

```
numpy
torch
torchvision
```

## Usage

```bash
git clone https://github.com/yuta0306/SRAugmentation.git
```

```python
# Path
import sys
sys.path.append('SRAugmentation')

from SRAugmentation import *

LR = torch.randn(3, 512, 512)  # <- Load low-resolution image
HR = torch.randn(3, 512, 512)  # <- Load high-resolution image
refLR = torch.randn(3, 128, 128)  # <- Load low-resolution image for reference
refHR = torch.randn(3, 512, 512)  # <- Load high-resolution image for reference

transforms = OneOf([
    Blend(),
    CutBlur(expand=True),
    CutMix(),
    CutMixup(),
    Cutout(),
    Mixup(),
    RGBPermutation(),
    Identity(),
])

LR, HR = transforms(LR, HR, refLR, refHR)
```

## License

> cf. @solafune(https://solafune.com) コンテストの参加以外を目的とした利用及び商用利用は禁止されています。商用利用・その他当コンテスト以外で利用したい場合はお問い合わせください。(https://solafune.com)

> cf. @solafune(https://solafune.com) Use for any purpose other than participation in the competition or commercial use is prohibited. If you would like to use them for any of the above purposes, please contact us.