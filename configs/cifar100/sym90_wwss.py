# model
num_classes = 100
model = dict(type='PreResNet', depth=18, num_classes=num_classes)

# TODO
noise_mode, noise_ratio = 'sym', 0.9  # ['asym', 'sym']
lam_u, lam_p = 150, 1.0
thresh_mode, p_margin, p_thresh = 'single', 0.1, 0.6  # ['cross', 'single']; if 'cross', set p_margin; TODO
# relabel
retype = 'all'  # ['all', 'unlabel']
t_relabel, re_epochs = [0.3], [101]
label_path = './datasets/labels/cifar100/all/sym90.npy'

# data
loss = dict(
    train=dict(type='SemiLoss') if lam_u > 0 else dict(type='SmoothCE'),
    test=dict(type='CrossEntropyLoss')
)
root = './data'
T_sharpen = 0.5
alpha = 4

# dataset
batch_size = 64
warmup_batch_size = 128
r_unlabel = 2
num_workers = 16
mean = (0.5071, 0.4867, 0.4408)
std = (0.2675, 0.2565, 0.2761)
data = dict(
    gt_labels='./datasets/labels/cifar100/gtlabels.npy',
    warmup=dict(
        ds_dict=dict(
            type='noisy_cifar100',
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
            type='noisy_cifar100',
            root=root,
            train=True,
            mode='eval_train',
            label_path=label_path,
        ),
        trans_dict=dict(type='cifar_test', mean=mean, std=std)
    ),
    relabel=dict(
        ds_dict=dict(
            type='noisy_cifar100',
            root=root,
            train=True,
            mode='relabel',
        ),
        trans_dict=dict(type='cifar_test', mean=mean, std=std)
    ),
    label=dict(
        ds_dict=dict(
            type='noisy_cifar100',
            root=root,
            train=True,
            mode='label',
            label_path=label_path,
        ),
        trans_dict=dict(
            type='CIFARMultiView',
            views='wwss', aug='auto',
            mean=mean, std=std,
        )
    ),
    unlabel=dict(
        ds_dict=dict(
            type='noisy_cifar100',
            root=root,
            train=True,
            mode='unlabel',
        ),
        trans_dict=dict(
            type='CIFARMultiView',
            views='wwss', aug='auto',
            mean=mean, std=std,
        )
    ),
    test=dict(
        ds_dict=dict(
            type='noisy_cifar100',
            root=root,
            train=False,
            mode='test',
        ),
        trans_dict=dict(type='cifar_test', mean=mean, std=std)
    ),
)

# training optimizer & scheduler
rampup_epochs = 16
warmup_epochs = 30
epochs = 400
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=5e-4)
lr_updater = dict(
    type='MultiStepLrUpdater',
    warmup='linear', warmup_steps=warmup_epochs, warmup_ratio=0.1,
    milestones=[300], gamma=0.1,
)

# log & save
log_interval = 100
save_interval = 100
test_interval = 1
work_dir = None  # rewritten by args
resume = None
load = None
port = 10001
