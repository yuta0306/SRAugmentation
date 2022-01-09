import numpy as np
import torch
import torchvision.transforms as T

from typing import Tuple

class CutBlur(object):
    """
    CutBlur
    低解像度画像と高解像度画像の一部分を入れ替える．

    Attributes
    ----------
    p : float
        CutBlurを実行する確率．
    alpha : float
        画像をくり抜くサイズのバイアス．
    expand : bool
        LRとHRの入力画像のサイズが異なる時，
        `expand=True`ならば，出力画像のサイズを揃える．
        `expand=False`ならば，LRの出力サイズを入力サイズに戻す．
    """
    def __init__(self, p: float = 1.0, alpha: float = 0.7, expand: bool = False):
        """
        Parameters
        ----------
        p : float
            CutBlurを実行する確率．
        alpha : float
            画像をくり抜くサイズのバイアス．
        expand : bool
            LRとHRの入力画像のサイズが異なる時，
            `expand=True`ならば，出力画像のサイズを揃える．
            `expand=False`ならば，LRの出力サイズを入力サイズに戻す．
        """
        self.p = p
        self.alpha = alpha
        self.expand = expand

    def __call__(self, LR: torch.Tensor, HR: torch.Tensor)\
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        LR : torch.Tensor
            低解像度画像
        HR : torch.Tensor
            高解像度画像

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            低解像度画像，高解像度画像
        """
        if self.alpha <= 0 or np.random.rand(1) >= self.p:
            return LR, HR

        LR_size = LR.size()[1:]
        HR_size = HR.size()[1:]
        if LR_size != HR_size:
            LR = T.Resize(HR_size, interpolation=T.InterpolationMode.NEAREST)(LR)

        cut_ratio = np.random.randn() * 0.01 + self.alpha

        h, w = HR_size
        ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)
        cy = np.random.randint(0, h-ch+1)
        cx = np.random.randint(0, w-cw+1)

        # apply CutBlur to inside or outside
        if np.random.random() > 0.5:
            LR[..., cy:cy+ch, cx:cx+cw] = HR[..., cy:cy+ch, cx:cx+cw]
        else:
            LR_aug = HR.clone()
            LR_aug[..., cy:cy+ch, cx:cx+cw] = LR[..., cy:cy+ch, cx:cx+cw]
            LR = LR_aug

        if self.expand:
            return LR, HR
        LR = T.Resize(LR_size)(LR)
        return LR, HR