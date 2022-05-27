import random
import numpy as np
import torch
import os
import re


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(gpuid: str) -> torch.device:
    if gpuid and torch.cuda.is_available():
        assert re.fullmatch(r"[0-7](,[0-7])*", gpuid) is not None, "invalid way to specify gpuid"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
        device = torch.device(f"cuda:{gpuid}")
    else:
        device = torch.device("cpu")

    return device