import numpy as np
import torch
import torchvision.transforms as T

from typing import Tuple, Optional

class Blend(object):
    """
    Blend
    連続一様分布に従いサンプルされた色を混ぜる．

    Attributes
    ----------
    p : float
        Blendを実行する確率．
    alpha : float
        元画像と混合する色の割合のうち，元画像の割合の最小値．
    rgb_range : float
        RGBがとる値の最大値．
        `1.0`か`255.0`を想定．
    """
    def __init__(self, p: float = 1.0, alpha: float = 0.6, rgb_range: float = 1.0):
        """
        Parameters
        ----------
        p : float
            Blendを実行する確率．
        alpha : float
            元画像と混合する色の割合のうち，元画像の割合の最小値．
        rgb_range : float
            RGBがとる値の最大値．
            `1.0`か`255.0`を想定．

        Raises
        ------
        ValueError
            rgb_rangeが`1.0`か`255.0`でない．
        """
        if rgb_range not in (1.0, 255.0):
            raise ValueError
        self.p = p
        self.alpha = alpha
        self.rgb_range = rgb_range
    
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

        c = torch.empty((3, 1, 1), device=LR.device).uniform_(0, self.rgb_range)
        rLR = c.repeat((1, LR.size(1), LR.size(2)))
        rHR = c.repeat((1, HR.size(1), HR.size(2)))

        v = np.random.uniform(self.alpha, 1)
        LR = v * LR + (1-v) * rLR
        HR = v * HR + (1-v) * rHR

        return LR, HR