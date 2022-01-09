import numpy as np
import torch
import torchvision.transforms as T

from typing import Tuple, Optional

class Identity(object):
    """
    Identity
    何も処理をしないクラス
    """
    def __init__(self):
        pass
    
    def __call__(self, *args):
        return args