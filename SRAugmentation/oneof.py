from mixup import Mixup
from cutmix import CutMix
from cutmixup import CutMixup

import numpy as np
import torch
import torchvision.transforms as T

from typing import List, Tuple, Optional

class OneOf(object):
    """
    OneOf
    augmentationのリストの中から，一つだけ実行するクラス

    Attributes
    ----------
    augs : List[object]
        augmentationのクラスを格納したリスト
        i.e) [Blend(), Cutup(), CutBlur(expand=True)]
    probs : Optional[List[float]]
        augsに格納された各augmentationが選ばれる確率のリスト
    """
    def __init__(self, augs: List[object], probs: Optional[List[float]] = None):
        """
        Parameters
        ----------
        augs : List[object]
            augmentationのクラスを格納したリスト
            i.e) [Blend(), Cutup(), CutBlur(expand=True)]
        probs : Optional[List[float]]
            augsに格納された各augmentationが選ばれる確率のリスト
        """
        self.augs = augs
        self.probs = probs

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
        idx = np.random.choice(len(self.augs), p=self.probs)
        aug = self.augs[idx]

        if isinstance(aug, (Mixup, CutMixup, CutMix)):
            LR, HR = aug(LR, HR, refLR, refHR)
        else:
            LR, HR = aug(LR, HR)

        return LR, HR