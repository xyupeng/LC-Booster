from .base_lr_updater import BaseLrUpdater
import numpy as np


class MultiStepLrUpdater(BaseLrUpdater):
    """
        decay at given steps.
    """
    def __init__(self, milestones=[], gamma=0.1, **kwargs):
        assert isinstance(milestones, (tuple, list))
        self.milestones = milestones
        self.gamma = gamma
        super().__init__(**kwargs)

    def get_lr(self, base_lr, cur_step, steps):
        num_steps = np.sum(cur_step >= np.asarray(self.milestones))
        lr = base_lr * (self.gamma ** num_steps)
        return lr
