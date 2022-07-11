from torchvision import transforms
from .randaugment import RandAugmentMC
from .autoaugment import CIFAR10Policy


def cifar_test(mean, std):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform


class CIFARMultiView(object):
    '''
        return multiple augmented views. (4 views by default)
    '''
    def __init__(self, mean, std, views='wwss', aug='auto', padding_mode='reflect'):
        assert all(v in 'ws' or v.isdigit() for v in views)  # ww01
        assert aug in ['rand', 'auto']
        self.views = views

        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=4, padding_mode=padding_mode),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        if aug == 'auto':
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=4, padding_mode=padding_mode),
                CIFAR10Policy(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:  # 'rand'
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=4, padding_mode=padding_mode),
                RandAugmentMC(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    def __call__(self, x):
        ret_views = []
        for v in self.views:
            if v == 'w':
                ret_views.append(self.weak(x))
            elif v == 's':
                ret_views.append(self.strong(x))
            else:
                idx = int(v)
                ret_views.append(ret_views[idx])
        if len(ret_views) > 1:
            return ret_views
        else:
            return ret_views[0]
