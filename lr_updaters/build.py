import lr_updaters


def build_lr_updater(cfg, optimizer):
    cfg_args = cfg.copy()
    func_name = cfg_args.pop('type')
    return lr_updaters.__dict__[func_name](optimizer=optimizer, **cfg_args)
