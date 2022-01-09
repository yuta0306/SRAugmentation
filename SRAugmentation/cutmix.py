import numpy as np
import torch
import torchvision.transforms as T

from typing import Tuple, Optional

class CutMix(object):
    """
    CutMix
    異なる画像をランダムなサイズで差し込む

    Attributes
    ----------
    p : float
        CutMixを実行する確率．
    alpha : float
        画像をくり抜くサイズのバイアス．
    """
    def __init__(self, p: float = 1.0, alpha: float = 0.7):
        """
        Parameters
        ----------
        p : float
            CutMixを実行する確率．
        alpha : float
            画像をくり抜くサイズのバイアス．
        """
        self.p = p
        self.alpha = alpha

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
            差し込む低解像度画像
        refHR : torch.Tensor
            差し込む高解像度画像

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            低解像度画像，高解像度画像
        """
        if np.random.rand(1) >= self.p:
            return LR, HR

        scale = HR.size(1) // LR.size(1)
        cut_ratio = np.random.randn() * 0.01 + self.alpha
        h, w = LR.size()[1:]
        ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)

        fcy = np.random.randint(0, h-ch+1)
        fcx = np.random.randint(0, w-cw+1)
        tcy, tcx = fcy, fcx

        LR[:, tcy:tcy+ch, tcx:tcx+cw] = refLR[:, fcy:fcy+ch, fcx:fcx+cw]
        HR[:, tcy*scale:tcy*scale+ch*scale, tcx*scale:tcx*scale+cw*scale] = refHR[:, fcy*scale:fcy*scale+ch*scale, fcx*scale:fcx*scale+cw*scale]
        
        return LR, HR