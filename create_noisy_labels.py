import argparse
import numpy as np
import json
import os
from utils.util import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', type=str, choices=['cifar10', 'cifar100'])
    parser.add_argument('--type', type=str, choices=['sym', 'asym'])
    parser.add_argument('--ratio', type=float)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def create_noisy_labels(args):
    num_classes = int(args.ds[5:])
    gt_labels = np.load(f'./datasets/labels/{args.ds}/gtlabels.npy')

    len_ds = 50000
    r = args.ratio  # noise ratio
    noise_samples = int(len_ds * r)

    if args.type == 'asym':
        assert num_classes == 10
        asym_map = {0: 0, 1: 1, 2: 0, 3: 5, 4: 7, 5: 3, 6: 6, 7: 7, 8: 8, 9: 1}  # class transition for asymmetric noise

    noise_idx = np.random.choice(len_ds, noise_samples, replace=False)
    noisy_labels = gt_labels.copy()

    if args.type == 'sym':
        noisy_labels[noise_idx] = np.random.randint(0, num_classes, noise_samples)
    else:
        flip_labels = [asym_map[y] for y in gt_labels[noise_idx]]
        noisy_labels[noise_idx] = np.array(flip_labels)

    # check label acc
    label_acc = (noisy_labels == gt_labels).astype(float).mean() * 100
    print(f'label acc: {label_acc:.3f}')

    indices = np.arange(len_ds)
    indices_labels = np.stack([indices, noisy_labels])
    save_path = f'./datasets/labels/{args.ds}/all/sym{int(r*100)}_seed{args.seed}'
    assert not os.path.isfile(save_path + '.npy')
    np.save(save_path, indices_labels)


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    create_noisy_labels(args)
