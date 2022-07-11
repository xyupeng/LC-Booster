import models
import torchvision.models as tvmodels


def build_model(cfg):
    args = cfg.copy()
    name = args.pop('type')
    if hasattr(tvmodels, name):
        model = getattr(tvmodels, name)(**args)
    else:
        model = models.__dict__[name](**args)
    return model
