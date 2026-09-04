import os
import random
import numpy as np
import torch

def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mkdir_if_not_exist(path: str):
    """文件夹不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)