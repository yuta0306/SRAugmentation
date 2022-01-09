import numpy as np
import torch
import torchvision.transforms as T

from typing import Tuple, Optional

class RGBPermutation(object):
    """
    RGBPermutation
    RGBをランダムで入れ替える

    Attributes
    ----------
    p : float
        RGBPermutationを実行する確率．
    """
    def __init__(self, p: float = 1.0):
        """
        Parameters
        ----------
        p : float
            RGBPermutationを実行する確率．
        """
        self.p = p

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
        if np.random.rand(1) >= self.p:
            return LR, HR
        
        perm = np.random.permutation(3)
        LR = LR[perm]
        HR = HR[perm]

        return LR, HR