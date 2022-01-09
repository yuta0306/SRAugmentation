import numpy as np
import torch
import torchvision.transforms as T

from typing import Tuple, Optional

class CutMixup(object):
    """
    CutMixup
    CutMix & Mixup

    Attributes
    ----------
    mixup_p : float
        Mixupを実行する確率．
    cutmix_p : float
        CutMixを実行する確率．
    mixup_alpha : float
        Mixupにおける，ベータ分布におけるalpha及びbeta
    cutmix_alpha : float
        CutMixにおける，画像をくり抜くサイズのバイアス．
    """
    def __init__(self, mixup_p: float = 1.0, cutmix_p: float = 1.0,
                 mixup_alpha: float = 1.2, cutmix_alpha: float = 0.7):
        """
        Parameters
        ----------
        mixup_p : float
            Mixupを実行する確率．
        cutmix_p : float
            CutMixを実行する確率．
        mixup_alpha : float
            Mixupにおける，ベータ分布におけるalpha及びbeta
        cutmix_alpha : float
            CutMixにおける，画像をくり抜くサイズのバイアス．
        """
        self.mixup_p = mixup_p
        self.cutmix_p = cutmix_p
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

    def __call__(self, LR: torch.Tensor, HR: torch.Tensor, refLR: torch.Tensor, refHR: torch.Tensor)\
        -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        LR : torch.Tensor
            低解像度画像
        HR : torch.Tensor
            高解像度画像
        refLR : torch.Tensor
            混合，差し込みする低解像度画像
        refHR : torch.Tensor
            混合，差し込みする高解像度画像

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            低解像度画像，高解像度画像
        """
        if np.random.rand(1) >= self.cutmix_p:
            return LR, HR

        scale = HR.size(1) // LR.size(1)
        cut_ratio = np.random.randn() * 0.01 + self.cutmix_alpha
        h, w = LR.size()[1:]
        ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)

        fcy = np.random.randint(0, h-ch+1)
        fcx = np.random.randint(0, w-cw+1)
        tcy, tcx = fcy, fcx

        v = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        if self.mixup_alpha <= 0 or np.random.rand(1) >= self.mixup_p:
            LR_aug = refLR
            HR_aug = refHR
        else:
            LR_aug = v * LR + (1-v) * refLR
            HR_aug = v * HR + (1-v) * refHR
        
        # apply mixup to inside or outside
        if np.random.random() > 0.5:
            LR[:, tcy:tcy+ch, tcx:tcx+cw] = LR_aug[:, fcy:fcy+ch, fcx:fcx+cw]
            HR[:, tcy*scale:tcy*scale+ch*scale, tcx*scale:tcx*scale+cw*scale] = HR_aug[:, fcy*scale:fcy*scale+ch*scale, fcx*scale:fcx*scale+cw*scale]
        else:
            LR_aug[:, tcy:tcy+ch, tcx:tcx+cw] = LR[:, fcy:fcy+ch, fcx:fcx+cw]
            HR_aug[:, tcy*scale:tcy*scale+ch*scale, tcx*scale:tcx*scale+cw*scale] = HR[:, fcy*scale:fcy*scale+ch*scale, fcx*scale:fcx*scale+cw*scale]
            LR, HR = LR_aug, HR_aug
        
        return LR, HR