import numpy as np
import torch
import torchvision.transforms as T

from typing import Tuple, Optional

class Cutout(object):
    """
    Cutout
    画像を一定確率でマスク．
    提案論文では，PSNRが多少増加する程度であまり効果は高くないとのこと．

    Attributes
    ----------
    p : float
        Cutoutを実行する確率．
    alpha : float
        Cutoutを実行した時，LR画像をランダムでマスクする割合．
    """
    def __init__(self, p: float = 1.0, alpha: float = 0.001):
        """
        Parameters
        ----------
        p : float
            Cutoutを実行する確率．
        alpha : float
            Cutoutを実行した時，LR画像をランダムでマスクする割合．
        """
        self.p = p
        self.alpha = alpha
    
    def __call__(self, LR: torch.Tensor, HR: Optional[torch.Tensor] = None)\
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
            if HR is None:
                return LR
            return LR, HR

        LR_size = LR.size()[1:]
        mask = np.random.choice([0.0, 1.0], size=LR_size, p=[self.alpha, 1-self.alpha])
        mask = torch.tensor(mask, dtype=torch.float32, device=LR.device)
        LR = LR * mask

        if HR is None:
            return LR
        return LR, HR