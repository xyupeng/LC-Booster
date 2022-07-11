import math
from .base_lr_updater import BaseLrUpdater


class CosineLrUpdater(BaseLrUpdater):
    def __init__(self, min_lr=None, min_lr_ratio=None, **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super(CosineLrUpdater, self).__init__(**kwargs)

    def get_lr(self, base_lr, cur_step, steps):
        if self.min_lr_ratio is not None:
            min_lr = base_lr * self.min_lr_ratio
        else:
            min_lr = self.min_lr
        lr = min_lr + (base_lr - min_lr) * (1 + math.cos(math.pi * cur_step / steps)) / 2
        return lr
