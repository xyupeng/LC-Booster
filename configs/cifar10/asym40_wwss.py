# model
num_classes = 10
model = dict(type='PreResNet', depth=18, num_classes=num_classes)

# TODO
noise_mode, noise_ratio = 'asym', 0.4  # ['asym', 'sym']
lam_u, lam_p = 0., 1.0
thresh_mode, p_thresh = 'single', 0.5  # ['cross', 'single']; if 'cross', set p_margin; TODO
# relabel
retype = 'all'  # ['all', 'unlabel']
t_relabel, re_epochs = [0.6], [101]
label_path = './datasets/labels/cifar10/all/asym40.npy'

# data
loss = dict(
    train=dict(type='SemiLoss') if lam_u > 0 else dict(type='SmoothCE'),
    test=dict(type='CrossEntropyLoss')
)
root = './data'
T_sharpen = 0.5
alpha = 4

# dataset
batch_size = 128
warmup_batch_size = 128
r_unlabel = 1
num_workers = 16
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
data = dict(
    gt_labels='./datasets/labels/cifar10/gtlabels.npy',
    warmup=dict(
        ds_dict=dict(
            type='noisy_cifar10',
            root=root,
            train=True,
            mode='warmup',
            label_path=label_path,
        ),
        trans_dict=dict(
            type='CIFARMultiView',
            views='w', mean=mean, std=std
        )
    ),
    eval_train=dict(
        ds_dict=dict(
            type='noisy_cifar10',
            root=root,
            train=True,
            mode='eval_train',
            label_path=label_path,
        ),
        trans_dict=dict(type='cifar_test', mean=mean, std=std)
    ),
    relabel=dict(
        ds_dict=dict(
            type='noisy_cifar10',
            root=root,
            train=True,
            mode='relabel',
        ),
        trans_dict=dict(type='cifar_test', mean=mean, std=std)
    ),
    label=dict(
        ds_dict=dict(
            type='noisy_cifar10',
            root=root,
            train=True,
            mode='label',
            label_path=label_path,
        ),
        trans_dict=dict(
            type='CIFARMultiView',
            views='wwss', aug='rand',
            mean=mean, std=std,
        )
    ),
    unlabel=dict(
        ds_dict=dict(
            type='noisy_cifar10',
            root=root,
            train=True,
            mode='unlabel',
        ),
        trans_dict=dict(
            type='CIFARMultiView',
            views='wwss', aug='rand',
            mean=mean, std=std,
        )
    ),
    test=dict(
        ds_dict=dict(
            type='noisy_cifar10',
            root=root,
            train=False,
            mode='test',
        ),
        trans_dict=dict(type='cifar_test', mean=mean, std=std)
    ),
)

# training optimizer & scheduler
rampup_epochs = 16
warmup_epochs = 10
epochs = 300
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=5e-4)
lr_updater = dict(
    type='MultiStepLrUpdater',
    warmup=None, warmup_steps=warmup_epochs, warmup_ratio=1.0,
    milestones=[250], gamma=0.1,
)

# log & save
log_interval = 100
save_interval = 100
test_interval = 1
work_dir = None  # rewritten by args
resume = None
load = None
port = 10001
