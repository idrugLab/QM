# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch


# -----------------------------------------------------------------------------
# Set seed for random, numpy, torch, cuda.
# -----------------------------------------------------------------------------
def seed_set(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# -----------------------------------------------------------------------------
# Checkpoint loading and saving.
# -----------------------------------------------------------------------------
class ThreeDEvaluator:
    def __init__(self):
        pass

    def eval(self, input_dict):
        assert('y_pred' in input_dict)
        assert('y_true' in input_dict)

        y_pred, y_true = input_dict['y_pred'], input_dict['y_true']

        assert((isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray))
                or
                (isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor)))
        assert(y_true.shape == y_pred.shape)

        if isinstance(y_true, torch.Tensor):
            return {'mae': torch.mean(torch.abs(y_pred - y_true)).cpu().item()}
        else:
            return {'mae': float(np.mean(np.absolute(y_pred - y_true)))}
