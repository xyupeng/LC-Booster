# Adapted from mmcv/runner/hooks/lr_updater.py


class BaseLrUpdater:
    """ Base LR Scheduler. All lr_updater should inherit this class.
        Currently support: 'lr_scale'

        0 <= cur_step < steps; Might be float or int
        warmup period: cur_step < self.warmup_steps
        regular_lr period: cur_step >= self.warmup_steps

    Args:
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        warmup_steps(int): The number of steps that warmup lasts
        warmup_ratio (float): LR used at the beginning of warmup equals to
            warmup_ratio * base_lr
    """

    def __init__(self,
                 optimizer,
                 warmup=None,
                 warmup_steps=0,
                 warmup_ratio=0.1):
        assert warmup in [None, 'const', 'linear', 'exp']
        if warmup is not None:
            assert warmup_steps > 0 and 0 <= warmup_ratio <= 1.0

        self.optimizer = optimizer
        self.warmup = warmup
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio

        # each param_group is a dict
        for group in optimizer.param_groups:
            group.setdefault('base_lr', group['lr'])

    def get_lr(self, base_lr, cur_step, steps):
        raise NotImplementedError

    def get_warmup_lr(self, cur_step, regular_lr):
        if self.warmup == 'const':
            warmup_lr = regular_lr * self.warmup_ratio
        elif self.warmup == 'linear':
            k = (1 - cur_step / self.warmup_steps) * (1 - self.warmup_ratio)
            warmup_lr = regular_lr * (1 - k)
        elif self.warmup == 'exp':
            k = self.warmup_ratio ** (1 - cur_step / self.warmup_steps)
            warmup_lr = regular_lr * k
        else:
            raise ValueError(self.warmup)
        return warmup_lr

    def adjust_lr(self, cur_step, steps):
        """
            0 <= cur_step < steps; Might be float or int
            warmup period: cur_step < self.warmup_steps,
            regular_lr period: cur_step >= self.warmup_steps
        """
        for param_group in self.optimizer.param_groups:
            regular_lr = self.get_lr(param_group['base_lr'], cur_step, steps)
            if self.warmup is not None and cur_step < self.warmup_steps:
                lr = self.get_warmup_lr(cur_step, regular_lr)
            else:
                lr = regular_lr

            if 'lr_scale' in param_group.keys():
                param_group['lr'] = lr * param_group['lr_scale']
            else:
                param_group['lr'] = lr
