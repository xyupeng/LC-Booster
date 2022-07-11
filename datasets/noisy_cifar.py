from .build import build_dataset
from torchvision.datasets import CIFAR10, CIFAR100
import torch
import numpy as np
from PIL import Image
import copy


class noisy_cifar10(CIFAR10):
    def __init__(self, root, train=True, transform=None, mode='label',
                 label_path=None, indices=None, probs=None, psl=None, **kwargs):
        '''
            label_path: .npy file with indices and labels
            indices: ndarray; sample indices in the original CIFAR10
            probs: only required when mode == 'label'; len(probs) == len(indices)
            ps: list or ndarray; pseudo labels;

            if not None, len(psl) == len(probs) == 50000
            len(indices) == len(self.data)
        '''
        assert mode in ['warmup', 'eval_train', 'relabel', 'label', 'unlabel', 'test']
        super().__init__(root, train=train, transform=transform, **kwargs)
        self.mode = mode

        # load noisy data/ps label if specified
        if label_path:
            noisy_indices, noisy_labels = np.load(label_path)
            self.data = self.data[noisy_indices]
            self.targets = noisy_labels.tolist()
        if psl is not None:
            if isinstance(psl, np.ndarray):
                self.targets = psl.tolist()
            else:
                self.targets = psl  # list

        self.indices = np.arange(len(self.data))  # assume always len(self.data) == 50000
        if self.mode == 'label':
            self.data = self.data[indices]
            self.targets = [self.targets[idx] for idx in indices]
            # self.probs = probs  # huge bug!!!!
            self.probs = probs[indices]
            self.indices = indices
        elif self.mode == 'unlabel':
            self.data = self.data[indices]
            self.indices = indices
        elif self.mode == 'relabel':
            if indices is not None:
                self.data = self.data[indices]
                self.indices = indices

    def __getitem__(self, index):
        if self.mode == 'warmup' or self.mode == 'eval_train' or self.mode == 'test':
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target
        elif self.mode == 'relabel':
            img = self.data[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, self.indices[index]
        elif self.mode == 'label':
            img, target, prob = self.data[index], self.targets[index], self.probs[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, prob
        elif self.mode == 'unlabel':
            img = self.data[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img
        else:
            raise ValueError


class noisy_cifar100(CIFAR100):
    def __init__(self, root, train=True, transform=None, mode='label',
                 label_path=None, indices=None, probs=None, psl=None, **kwargs):
        '''
            label_path: .npy file with indices and labels
            indices: ndarray; sample indices in the original CIFAR10
            probs: only required when mode == 'label'; len(probs) == len(indices)
            ps: list or ndarray; pseudo labels;

            if not None, len(psl) == len(probs) == 50000
            len(indices) == len(self.data)
        '''
        assert mode in ['warmup', 'eval_train', 'relabel', 'label', 'unlabel', 'test']
        super().__init__(root, train=train, transform=transform, **kwargs)
        self.mode = mode

        # load noisy data/ps label if specified
        if label_path:
            noisy_indices, noisy_labels = np.load(label_path)
            self.data = self.data[noisy_indices]
            self.targets = noisy_labels.tolist()
        if psl is not None:
            if isinstance(psl, np.ndarray):
                self.targets = psl.tolist()
            else:
                self.targets = psl  # list

        self.indices = np.arange(len(self.data))  # assume always len(self.data) == 50000
        if self.mode == 'label':
            self.data = self.data[indices]
            self.targets = [self.targets[idx] for idx in indices]
            # self.probs = probs  # huge bug!!!!
            self.probs = probs[indices]
            self.indices = indices
        elif self.mode == 'unlabel':
            self.data = self.data[indices]
            self.indices = indices
        elif self.mode == 'relabel':
            if indices is not None:
                self.data = self.data[indices]
                self.indices = indices

    def __getitem__(self, index):
        if self.mode == 'warmup' or self.mode == 'eval_train' or self.mode == 'test':
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target
        elif self.mode == 'relabel':
            img = self.data[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, self.indices[index]
        elif self.mode == 'label':
            img, target, prob = self.data[index], self.targets[index], self.probs[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, prob
        elif self.mode == 'unlabel':
            img = self.data[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img
        else:
            raise ValueError


def build_cifar_loader(cfg, mode, indices=None, probs=None, psl=None):
    assert mode in ['warmup', 'eval_train', 'relabel', 'label', 'unlabel', 'test']
    if mode == 'warmup':
        warmup_set = build_dataset(cfg.data.warmup)
        warmup_loader = torch.utils.data.DataLoader(
            warmup_set, batch_size=cfg.warmup_batch_size, shuffle=True,
            num_workers=cfg.num_workers, drop_last=True
        )
        return warmup_loader
    elif mode == 'eval_train':
        eval_train_cfg = copy.deepcopy(cfg.data.eval_train)
        eval_train_cfg.ds_dict.psl = psl
        eval_train_set = build_dataset(eval_train_cfg)
        eval_train_loader = torch.utils.data.DataLoader(
            eval_train_set, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, drop_last=False
        )
        return eval_train_loader
    elif mode == 'relabel':
        relabel_cfg = copy.deepcopy(cfg.data.relabel)
        relabel_cfg.ds_dict.indices = indices
        relabel_set = build_dataset(relabel_cfg)
        relabel_loader = torch.utils.data.DataLoader(
            relabel_set, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, drop_last=False
        )
        return relabel_loader
    elif mode == 'label':
        label_cfg = copy.deepcopy(cfg.data.label)
        label_cfg.ds_dict.indices = indices
        label_cfg.ds_dict.probs = probs
        label_cfg.ds_dict.psl = psl
        label_set = build_dataset(label_cfg)
        label_loader = torch.utils.data.DataLoader(
            label_set, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, drop_last=True
        )
        return label_loader
    elif mode == 'unlabel':
        unlabel_cfg = copy.deepcopy(cfg.data.unlabel)
        unlabel_cfg.ds_dict.indices = indices
        unlabel_set = build_dataset(unlabel_cfg)
        unlabel_loader = torch.utils.data.DataLoader(
            unlabel_set, batch_size=cfg.batch_size * cfg.r_unlabel, shuffle=True,
            num_workers=cfg.num_workers, drop_last=True
        )
        return unlabel_loader
    elif mode == 'test':
        test_set = build_dataset(cfg.data.test)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, drop_last=False
        )
        return test_loader
    else:
        raise ValueError


if __name__ == '__main__':
    pass
