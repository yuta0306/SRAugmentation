import numpy as np
import torch
import torchvision.transforms as T

from typing import Tuple, Optional

class Mixup(object):
    """
    Mixup
    ベータ分布に従って，異なる画像を混合する

    Attributes
    ----------
    p : float
        Mixupを実行する確率．
    alpha : float
        ベータ分布におけるalpha及びbeta
    """
    def __init__(self, p: float = 1.0, alpha: float = 1.2):
        """
        Parameters
        ----------
        p : float
            Mixupを実行する確率．
        alpha : float
            ベータ分布におけるalpha及びbeta
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
            混合する低解像度画像
        refHR : torch.Tensor
            混合する高解像度画像

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            低解像度画像，高解像度画像
        """
        if self.alpha <= 0 or np.random.rand(1) >= self.p:
            return LR, HR

        v = np.random.beta(self.alpha, self.alpha)
        
        LR = v * LR + (1-v) * refLR
        HR = v * HR + (1-v) * refHR

        return LR, HR