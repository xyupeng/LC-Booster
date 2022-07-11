### .txt
`all/xxx.txt` is used by Dataset `noisy_cifar10`

`labeled/xxx.txt` is used by Dataset `sub_noisy_cifar10` and `unlabel_cifar10`

### .npy
`labeled/s5000_acc70.npy`: random 5000 samples, label_acc=70

`labeled/knn_s3000_acc77.npy`: 3000 samples after knn relabel, label_acc=77
```
gt_labels = np.load('gtlabels.npy')
indices, labels = np.load('s5000_acc70.npy')
```

