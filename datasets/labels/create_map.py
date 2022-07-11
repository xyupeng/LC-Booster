from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
import json
import os
from utils.util import set_seed


def save_gt():
    ds = CIFAR100(root='E:/data', train=True)
    len_ds = len(ds)

    labels = []
    for i in range(len_ds):
        _, label = ds[i]
        labels.append(label)

    labels = np.array(labels)
    np.save('./cifar100/gtlabels', labels)

    # # load
    # labels = np.load('./cifar10/gtlabels.npy')
    # print(labels[:100])

    # gt.txt
    outf = open('cifar100/gtlabels.txt', 'w')
    for l in labels:
        print(l, file=outf)


def create_labeled_set():
    ds = CIFAR10(root='E:/data', train=True)
    len_ds = len(ds)
    gtlabels = np.load('cifar10/gtlabels.npy')

    num_samples = 2456
    r = 0.0  # abs noise ratio; i.e., class 0 -> class[1-9]
    noise_samples = int(num_samples * r)

    labeled_idx = np.random.choice(len(ds), num_samples, replace=False)  # (num_samples, )
    print(labeled_idx[:10])
    labeled_labels = gtlabels[labeled_idx]  # (num_samples, )

    noise_idx = np.random.choice(num_samples, noise_samples, replace=False)  # (noise_samples, )
    for idx in noise_idx:
        new_label = np.random.randint(0, 9)
        new_label = new_label + int(new_label >= labeled_labels[idx])
        labeled_labels[idx] = new_label

    acc = int((1-r) * 100)
    idx_label = np.stack([labeled_idx, labeled_labels])
    save_path = f'./cifar10/labeled/s{num_samples}_acc{acc}'
    np.save(save_path, idx_label)
    # outf = open(f'./cifar10/labeled/s{num_samples}_acc{acc}.txt', 'w')
    # for idx, label in zip(labeled_idx, labeled_labels):
    #     print(idx, label, file=outf)

    # test label acc
    data = np.load(save_path + '.npy')
    indices = data[0]
    noisy_labels = data[1]
    gt_labels = gtlabels[indices]
    test_acc = np.mean(gt_labels == noisy_labels)
    print(f'Test label acc: {test_acc}')


def symmetric():
    # ds = CIFAR100(root='E:/data', train=True)
    gtlabels = np.load('./cifar100/gtlabels.npy')
    len_ds = len(gtlabels)

    r = 0.9  # noise ratio
    noise_samples = int(len_ds * r)

    noise_idx = np.random.choice(len_ds, noise_samples, replace=False)
    noise_labels = np.random.randint(0, 100, noise_samples)

    noise_idx, noise_labels = zip(*sorted(zip(noise_idx, noise_labels)))

    outf = open(f'./cifar10/all/sym{int(r*100)}_sorted.txt', 'w')
    for idx, label in zip(noise_idx, noise_labels):
        print(idx, label, file=outf)


def symmetric_all():
    num_classes = 10
    gt_labels = np.load(f'cifar{num_classes}/gtlabels.npy')

    len_ds = 50000
    r = 0.92  # noise ratio
    noise_samples = int(len_ds * r)

    noise_idx = np.random.choice(len_ds, noise_samples, replace=False)
    print(noise_idx[:5])

    noisy_labels = gt_labels.copy()
    noisy_labels[noise_idx] = np.random.randint(0, num_classes, noise_samples)
    label_acc = (noisy_labels == gt_labels).astype(float).mean() * 100
    print(f'label acc: {label_acc:.3f}')

    indices = np.arange(len_ds)
    indices_labels = np.stack([indices, noisy_labels])
    save_path = f'./cifar{num_classes}/all/sym{int(r*100)}'
    assert not os.path.isfile(save_path + '.npy')
    np.save(save_path, indices_labels)


def asym_all():
    # ds = CIFAR10(root='E:/data', train=True)
    # len_ds = len(ds)
    gt_labels = np.load('cifar10/gtlabels.npy')
    num_samples = len(gt_labels)

    asym_map = {0: 0, 1: 1, 2: 0, 3: 5, 4: 7, 5: 3, 6: 6, 7: 7, 8: 8, 9: 1}  # class transition for asymmetric noise
    r = 0.4  # noise ratio
    noise_samples = int(num_samples * r)

    noise_idx = np.random.choice(num_samples, noise_samples, replace=False)
    print(noise_idx[:5])

    noisy_labels = gt_labels.copy()
    flip_labels = [asym_map[y] for y in gt_labels[noise_idx]]
    noisy_labels[noise_idx] = np.array(flip_labels)

    label_acc = (noisy_labels == gt_labels).astype(float).mean() * 100
    print(f'label acc: {label_acc:.3f}')

    indices = np.arange(num_samples)
    indices_labels = np.stack([indices, noisy_labels])
    save_path = f'./cifar10/all/asym{int(r*100)}'
    assert not os.path.isfile(save_path + '.npy')
    np.save(save_path, indices_labels)

    # # sort and write
    # noise_idx, noise_labels = zip(*sorted(zip(noise_idx, noise_labels)))
    # outf = open(f'./cifar10/asym{int(r*100)}_sorted.txt', 'w')
    # for idx, label in zip(noise_idx, noise_labels):
    #     print(idx, label, file=outf)


def filter_easy():
    gt_labels = np.load('./cifar10/gtlabels.npy')

    epoch0 = np.load('cifar10/labeled/knn_s3223_acc76.npy')
    indices, labels = epoch0[0], epoch0[1]

    gt_labels = gt_labels[indices]
    mask = gt_labels == labels

    indices = indices[mask]
    labels = labels[mask]
    acc = mask.astype(float).mean()
    print(acc, len(indices))

    to_save = np.stack([indices, labels])
    np.save('./cifar10/labeled/easy_acc100', to_save)


def load_check_acc():
    gt_labels = np.load('./cifar10/gtlabels.npy')
    indices, labels = np.load('cifar10/all/asym40.npy')
    acc = (gt_labels[indices] == labels).astype(float).mean()

    print(len(indices), len(labels))
    print(acc)


def to_json():
    label_path = './cifar10/all/sym95.npy'
    save_path = './cifar10/all/sym95.json'
    assert not os.path.isfile(save_path)

    indices, labels = np.load(label_path)
    labels = labels.tolist()
    json.dump(labels, open(save_path, 'w'))


if __name__ == '__main__':
    seed = 42
    set_seed(42)

    symmetric_all()
    # load_check_acc()
    # to_json()
