import os
import argparse
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from sklearn.mixture import GaussianMixture
from torchnet.meter import AUCMeter

from utils.util import AverageMeter, accuracy, TrackMeter, interleave, de_interleave, set_seed
from utils.util import adjust_learning_rate, warmup_learning_rate, get_lr, count_params

from utils.config import Config, ConfigDict, DictAction
from losses import build_loss
from builder import build_optimizer
from models.build import build_model
from utils.util import format_time
from builder import build_logger
from datasets import build_cifar_loader
from lr_updaters import build_lr_updater


'''
x_w1, x_w2, u_w1, u_s1
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume', type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('--load', type=str, help='Load init weights for fine-tune (default: None)')
    parser.add_argument('--cfgname', help='specify log_file; for debug use')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='update the config; e.g., --cfg-options use_ema=True k1=a,b k2="[a,b]"'
                             'Note that the quotation marks are necessary and that no white space is allowed.')
    args = parser.parse_args()
    return args


def get_cfg(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        dirname = os.path.dirname(args.config).replace('configs', 'checkpoints', 1)
        filename = os.path.splitext(os.path.basename(args.config))[0]
        cfg.work_dir = os.path.join(dirname, filename)
    os.makedirs(cfg.work_dir, exist_ok=True)

    # cfgname
    if args.cfgname is not None:
        cfg.cfgname = args.cfgname
    else:
        cfg.cfgname = os.path.splitext(os.path.basename(args.config))[0]
    assert cfg.cfgname is not None

    # cfg.warm_cfg
    if hasattr(cfg, 'warm_cfg'):
        cfg.warm_cfg.warmup_to = get_lr(cfg.lr_cfg, cfg.warm_cfg.warm_epochs+1)

    # resume or load init weights
    if args.resume:
        cfg.resume = args.resume
    if args.load:
        cfg.load = args.load
    assert not (cfg.resume and cfg.load)

    # retype
    assert cfg.retype in ['all', 'unlabel']  # TODO

    # seed
    if args.seed is not None:
        cfg.seed = args.seed
    elif not hasattr(cfg, 'seed'):
        cfg.seed = 41
    set_seed(cfg.seed)

    return cfg


def load_weights(ckpt_path, model1, model2, optimizer1=None, optimizer2=None, resume=True):
    # load checkpoint
    assert os.path.isfile(ckpt_path)
    print(f'==> Loading checkpoint {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location='cuda')

    # load model & optimizer state_dict
    all_losses1, all_losses2 = [], []
    test_meter = TrackMeter()
    if resume:
        model1.load_state_dict(checkpoint['model1_state'])
        model2.load_state_dict(checkpoint['model2_state'])
        optimizer1.load_state_dict(checkpoint['optimizer1_state'])
        optimizer2.load_state_dict(checkpoint['optimizer2_state'])
        if 'cur_targets' in checkpoint.keys():
            global cur_targets
            cur_targets = checkpoint['cur_targets']
        if 'all_losses1' in checkpoint.keys():
            all_losses1 = checkpoint['all_losses1']
            all_losses2 = checkpoint['all_losses2']
            all_losses1 = [ls.cpu() for ls in all_losses1]
            all_losses2 = [ls.cpu() for ls in all_losses2]
        if 'test_meter' in checkpoint.keys():
            test_meter = checkpoint['test_meter']
        start_epoch = checkpoint['epoch'] + 1
    else:
        raise NotImplementedError
    print(f'==> Ckpt loaded (resume={resume}, start_epoch={start_epoch}).')
    return start_epoch, all_losses1, all_losses2, test_meter


def warmup(warm_loader, model, optimizer, epoch, logger, cfg):
    batch_time = AverageMeter()
    losses = AverageMeter()

    criterion = nn.CrossEntropyLoss().cuda()
    num_iters = len(warm_loader)

    model.train()
    t1 = end = time.time()
    for batch_idx, (inputs, labels) in enumerate(warm_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if cfg.noise_mode == 'asym':
            probs = torch.softmax(outputs, dim=1)
            neg_entropy = torch.mean(torch.sum(probs * probs.log(), dim=1))
            loss += neg_entropy

        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger
        if batch_idx % cfg.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch [{epoch}][{batch_idx}/{num_iters}] - '
                        f'Batch time: {batch_time.avg:.2f}, '
                        f'lr: {lr:.4f}, '
                        f'loss: {losses.avg:.3f}')

    t2 = time.time()
    epoch_time = format_time(t2 - t1)
    logger.info(f'Epoch [{epoch}] - train_time: {epoch_time}, '
                f'train_loss: {losses.avg:.3f}\n')


def test(test_loader, model1, model2, criterion, epoch, logger, writer):
    """validation"""
    logger.info(f'==> Test at Epoch [{epoch}]...')
    model1.eval()
    model2.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    time1 = time.time()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output1 = model1(images)
            output2 = model2(images)
            output = (output1 + output2) / 2
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

    # writer
    writer.add_scalar(f'Loss/test', losses.avg, epoch)
    writer.add_scalar(f'Acc/test', top1.avg, epoch)

    # logger
    time2 = time.time()
    epoch_time = format_time(time2 - time1)
    logger.info(f'test_time: {epoch_time}, '
                f'test_loss: {losses.avg:.3f}, '
                f'test_Acc@1: {top1.avg:.2f}')
    return top1.avg


def eval_train(eval_loader, model, all_losses, cfg):
    model.eval()
    losses = []
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss)
    losses = torch.cat(losses)
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    losses = losses.cpu()
    all_losses.append(losses)

    if cfg.noise_ratio >= 0.9:  # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_losses)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)

    # fit a two-component GMM to the loss
    # gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=cfg.get('gmm_tol', 1e-2), reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob


def relabel(relabel_loader, model1, model2, epoch, logger, cfg):
    '''
        no labels used in this function;
        predict pseudo label based on probs
    '''
    t1 = time.time()
    logger.info(f'Relabel at Epoch [{epoch}].')
    model1.eval()
    model2.eval()

    pred_labels = []
    pred_probs = []
    relabel_indices = []
    with torch.no_grad():
        for idx, (images, indices) in enumerate(relabel_loader):
            images = images.cuda()

            logits1 = model1(images)  # (b, C)
            logits2 = model2(images)  # (b, C)
            logits = (logits1 + logits2) / 2
            probs = torch.softmax(logits, dim=1)
            max_probs, max_targets = torch.max(probs, dim=1)

            pred_labels.append(max_targets)
            pred_probs.append(max_probs)
            relabel_indices.append(indices)

    pred_labels = torch.cat(pred_labels).cpu().numpy()
    pred_probs = torch.cat(pred_probs).cpu().numpy()
    relabel_indices = torch.cat(relabel_indices)

    global gt_noisy_mask
    global cur_targets
    global gt_labels
    relabel_idx = cfg.re_epochs.index(epoch)
    t_relabel = cfg.t_relabel[relabel_idx]
    mask = pred_probs >= t_relabel

    all_mask = np.zeros(len(cur_targets), dtype=bool)
    all_mask[relabel_indices] = mask
    cur_targets[all_mask] = pred_labels[mask]

    # eval_train_set.targets = cur_targets.tolist()
    gt_noisy_mask = gt_labels == cur_targets
    relabel_acc = (cur_targets[all_mask] == gt_labels[all_mask]).astype(float).mean() * 100
    logger.info(f'Relabel at Epoch [{epoch}]: relabel_acc={relabel_acc:.2f} '
                f'(num_relabel={mask.sum()}, thresh={t_relabel:.2f})')

    t2 = time.time()
    test_time = format_time(t2 - t1)
    logger.info(f'Total relabel time: {test_time}.\n')
    return pred_probs, pred_labels


def train(label_loader, unlabel_loader, model, model2, criterion, optimizer, epoch, logger, writer, cfg):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    t1 = end = time.time()

    labeled_train_iter = iter(label_loader)
    unlabeled_train_iter = iter(unlabel_loader)

    model.train()
    model2.eval()
    num_iters = len(label_loader)
    for batch_idx in range(num_iters):
        # fetch data
        try:
            (inputs_x_w1, inputs_x_w2), targets_x, w_x = next(labeled_train_iter)
        except:
            assert False
        try:
            (inputs_u_w1, inputs_u_s1) = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabel_loader)
            (inputs_u_w1, inputs_u_s1) = next(unlabeled_train_iter)
        batch_size = inputs_x_w1.size(0)

        # to cuda
        inputs_x_w1, inputs_x_w2 = inputs_x_w1.cuda(), inputs_x_w2.cuda()
        inputs_u_w1, inputs_u_s1 = inputs_u_w1.cuda(), inputs_u_s1.cuda()
        targets_x = torch.zeros(batch_size, cfg.num_classes).scatter_(1, targets_x.view(-1, 1), 1).cuda()
        w_x = w_x.view(-1, 1).cuda()

        # co-refinement and co-guessing
        with torch.no_grad():
            # label refinement of labeled samples
            outputs_x = model(inputs_x_w1)
            outputs_x2 = model(inputs_x_w2)

            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x * targets_x + (1 - w_x) * px
            ptx = px ** (1 / cfg.T_sharpen)  # temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

            # label co-guessing of unlabeled samples
            outputs_u_w11 = model(inputs_u_w1)
            outputs_u_w12 = model2(inputs_u_w1)

            pu = (torch.softmax(outputs_u_w11, dim=1) + torch.softmax(outputs_u_w12, dim=1)) / 2
            ptu = pu ** (1 / cfg.T_sharpen)

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

        # mixmatch forward
        lam = np.random.beta(cfg.alpha, cfg.alpha)
        lam = max(lam, 1 - lam)

        all_inputs = torch.cat([inputs_x_w1, inputs_x_w2, inputs_u_w1, inputs_u_s1], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        lam_u = cfg.lam_u
        if cfg.lam_u > 0:
            mixed_input = lam * input_a + (1 - lam) * input_b
            mixed_target = lam * target_a + (1 - lam) * target_b

            mixed_inputs = interleave(mixed_input, batch_size)
            logits = model(mixed_inputs)
            logits = de_interleave(logits, batch_size)

            # loss
            Lx, Lu = criterion(
                logits[:batch_size * 2], mixed_target[:batch_size * 2],
                logits[batch_size * 2:], mixed_target[batch_size * 2:]
            )
            cur_epoch = epoch - 1 + batch_idx / num_iters
            lam_u = cfg.lam_u * np.clip((cur_epoch - cfg.warmup_epochs) / cfg.rampup_epochs, 0., 1.)
            loss = Lx + lam_u * Lu
            losses_u.update(Lu.item())
        else:
            mixed_input = lam * input_a[:batch_size * 2] + (1 - lam) * input_b[:batch_size * 2]
            mixed_target = lam * target_a[:batch_size * 2] + (1 - lam) * target_b[:batch_size * 2]

            mixed_inputs = interleave(mixed_input, batch_size)
            logits = model(mixed_inputs)
            logits = de_interleave(logits, batch_size)

            Lx = criterion(logits, mixed_target)  # SmoothCE
            loss = Lx

        # penalty
        if cfg.lam_p > 0:
            prior = torch.ones(cfg.num_classes).cuda() / cfg.num_classes
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))
            loss += cfg.lam_p * penalty

        # update losses
        losses.update(loss.item())
        losses_x.update(Lx.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger
        if batch_idx % cfg.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch [{epoch}][{batch_idx}/{num_iters}] - '
                        f'Batch time: {batch_time.avg:.2f}, '
                        f'lr: {lr:.4f}, '
                        f'loss: {losses.avg:.3f}, '
                        f'loss_x: {losses_x.avg:.3f}, '
                        f'loss_u: {losses_u.avg:.3f}(lam_u={lam_u:.2f})')

    t2 = time.time()
    epoch_time = format_time(t2 - t1)
    logger.info(f'Epoch [{epoch}] - train_time: {epoch_time}, '
                f'train_loss: {losses.avg:.3f}\n')
    return losses.avg, losses_x.avg, losses_u.avg


def main():
    # args & cfg
    args = parse_args()
    cfg = get_cfg(args)  # modify cfg according to args
    cudnn.benchmark = True

    # write cfg
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.cfg')
    with open(log_file, 'a') as f:
        f.write(cfg.pretty_text)

    # logger
    logger = build_logger(cfg.work_dir, cfgname='train')
    writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, f'tensorboard'))

    '''
    # -----------------------------------------
    # build eval_train, test, warmup dataloader
    # -----------------------------------------
    '''
    warmup_loader = build_cifar_loader(cfg, mode='warmup')
    test_loader = build_cifar_loader(cfg, mode='test')
    logger.info(f'==> DataLoader built.')

    # init cur_targets
    global gt_labels, cur_targets, gt_noisy_mask
    gt_labels = np.load(cfg.data.gt_labels)
    cur_targets = np.array(warmup_loader.dataset.targets)
    gt_noisy_mask = gt_labels == cur_targets

    '''
    # -----------------------------------------
    # build model & optimizer
    # -----------------------------------------
    '''
    model1 = build_model(cfg.model)
    model2 = build_model(cfg.model)
    num_params = count_params(model1) / 1e6

    model1 = torch.nn.DataParallel(model1).cuda()
    model2 = torch.nn.DataParallel(model2).cuda()

    optimizer1 = build_optimizer(cfg.optimizer, model1.parameters())
    optimizer2 = build_optimizer(cfg.optimizer, model2.parameters())
    lr_updater1 = build_lr_updater(cfg.lr_updater, optimizer=optimizer1)
    lr_updater2 = build_lr_updater(cfg.lr_updater, optimizer=optimizer2)

    train_criterion = build_loss(cfg.loss.train).cuda()
    test_criterion = build_loss(cfg.loss.test).cuda()
    logger.info(f'==> Model built ({cfg.model.type}, params={num_params:.2f}M).')

    # resume or load init weights
    start_epoch = 1
    all_losses1, all_losses2 = [], []
    test_meter = TrackMeter()
    if cfg.resume:
        start_epoch, all_losses1, all_losses2, test_meter = \
            load_weights(cfg.resume, model1, model2, optimizer1, optimizer2, resume=True)  # load cur_targets
    elif cfg.load:
        load_weights(cfg.load, model1, model2, resume=False)

    '''
    # -----------------------------------------
    # warm up 
    # -----------------------------------------
    '''
    logger.info("==> Start training...")
    if start_epoch == 1:
        for epoch in range(1, cfg.warmup_epochs + 1):
            lr_updater1.adjust_lr(epoch - 1, steps=cfg.epochs)
            lr_updater2.adjust_lr(epoch - 1, steps=cfg.epochs)

            warmup(warmup_loader, model1, optimizer1, epoch, logger, cfg)
            warmup(warmup_loader, model2, optimizer2, epoch, logger, cfg)

            # test
            test_acc = test(test_loader, model1, model2, test_criterion, epoch, logger, writer)
            test_meter.update(test_acc, idx=epoch)
            logger.info(f'Best acc: {test_meter.max_val:.2f} (epoch={test_meter.max_idx}).')

            # save last warm epoch
            if epoch == cfg.warmup_epochs:
                save_path = os.path.join(cfg.work_dir, f'warm_{epoch}.pth')
                state_dict = {
                    'model1_state': model1.state_dict(),
                    'model2_state': model2.state_dict(),
                    'optimizer1_state': optimizer1.state_dict(),
                    'optimizer2_state': optimizer2.state_dict(),
                    'epoch': epoch,
                    'test_meter': test_meter
                }
                torch.save(state_dict, save_path)

    '''
    # -----------------------------------------
    # training
    # -----------------------------------------
    '''
    start_epoch = max(start_epoch, cfg.warmup_epochs + 1)
    for epoch in range(start_epoch, cfg.epochs + 1):
        '''
        # -----------------------------------------
        # Divide label/unlabel set
        # -----------------------------------------
        '''
        eval_train_loader = build_cifar_loader(cfg, mode='eval_train', psl=cur_targets)
        # prob, mask
        prob1 = eval_train(eval_train_loader, model1, all_losses1, cfg)
        prob2 = eval_train(eval_train_loader, model2, all_losses2, cfg)
        if cfg.thresh_mode == 'cross':
            mask1 = prob1 > cfg.p_thresh + cfg.p_margin * (epoch & 1)
            mask2 = prob2 > cfg.p_thresh + cfg.p_margin * (~epoch & 1)
        elif cfg.thresh_mode == 'single':
            mask1 = prob1 > cfg.p_thresh
            mask2 = prob2 > cfg.p_thresh
        else:
            raise ValueError

        # iou
        num_intersection = np.logical_and(mask1, mask2).sum()
        num_union = np.logical_or(mask1, mask2).sum()
        iou = num_intersection / num_union
        logger.info(f'Epoch [{epoch}] - mask1 mask2 iou: {iou:.3f} ({num_intersection}/{num_union})')

        # AUC
        auc_meter = AUCMeter()
        auc_meter.add(prob1, gt_noisy_mask)
        auc1, _, _ = auc_meter.value()
        auc_meter = AUCMeter()
        auc_meter.add(prob2, gt_noisy_mask)
        auc2, _, _ = auc_meter.value()
        logger.info(f'Epoch [{epoch}] - AUC_1: {auc1:.3f} (num_clean={mask1.sum()}), '
                    f'AUC_2={auc2:.3f} (num_clean={mask2.sum()})')

        '''
        # -----------------------------------------
        # Train 
        # -----------------------------------------
        '''
        lr_updater1.adjust_lr(epoch - 1, steps=cfg.epochs)
        lr_updater2.adjust_lr(epoch - 1, steps=cfg.epochs)
        # adjust_learning_rate(cfg.lr_cfg, optimizer1, epoch)
        # adjust_learning_rate(cfg.lr_cfg, optimizer2, epoch)

        label_indices = mask2.nonzero()[0]
        unlabel_indices = (~mask2).nonzero()[0]
        label_loader = build_cifar_loader(cfg, mode='label', indices=label_indices, probs=prob2, psl=cur_targets)
        unlabel_loader = build_cifar_loader(cfg, mode='unlabel', indices=unlabel_indices)
        train(label_loader, unlabel_loader, model1, model2, train_criterion, optimizer1,
              epoch, logger, writer, cfg)

        label_indices = mask1.nonzero()[0]
        unlabel_indices = (~mask1).nonzero()[0]
        label_loader = build_cifar_loader(cfg, mode='label', indices=label_indices, probs=prob1, psl=cur_targets)
        unlabel_loader = build_cifar_loader(cfg, mode='unlabel', indices=unlabel_indices)
        train(label_loader, unlabel_loader, model2, model1, train_criterion, optimizer2,
              epoch, logger, writer, cfg)

        '''
        # -----------------------------------------
        # After epoch hooks 
        # -----------------------------------------
        '''
        # save
        if epoch % cfg.save_interval == 0:
            model_path = os.path.join(cfg.work_dir, f'epoch_{epoch}.pth')
            state_dict = {
                'model1_state': model1.state_dict(),
                'model2_state': model2.state_dict(),
                'optimizer1_state': optimizer1.state_dict(),
                'optimizer2_state': optimizer2.state_dict(),
                'epoch': epoch,
                'cur_targets': cur_targets,
                'all_losses1': all_losses1,
                'all_losses2': all_losses2,
                'test_meter': test_meter
            }
            torch.save(state_dict, model_path)

        '''
        # -----------------------------------------
        # relabel all
        # -----------------------------------------
        '''
        if epoch in cfg.re_epochs:
            if cfg.retype == 'all':
                relabel_loader = build_cifar_loader(cfg, mode='relabel')
            elif cfg.retype == 'unlabel':
                common_unlabel_mask = ~(mask1 & mask2)
                common_indices = common_unlabel_mask.nonzero()[0]
                relabel_loader = build_cifar_loader(cfg, mode='relabel', indices=common_indices)
            else:
                raise ValueError
            relabel(relabel_loader, model1, model2, epoch, logger, cfg)

        # test
        if epoch % cfg.test_interval == 0:
            test_acc = test(test_loader, model1, model2, test_criterion, epoch, logger, writer)
            test_meter.update(test_acc, idx=epoch)
            logger.info(f'Best acc: {test_meter.max_val:.2f} (epoch={test_meter.max_idx}).')

    logger.info(f'Last 5 Acc: {test_meter.last(5):.2f}.')

    # save last
    model_path = os.path.join(cfg.work_dir, 'last.pth')
    state_dict = {
        'model1_state': model1.state_dict(),
        'model2_state': model2.state_dict(),
        'optimizer1_state': optimizer1.state_dict(),
        'optimizer2_state': optimizer2.state_dict(),
        'epoch': cfg.epochs,
        'cur_targets': cur_targets,
        'all_losses1': all_losses1,
        'all_losses2': all_losses2,
        'test_meter': test_meter
    }
    torch.save(state_dict, model_path)


if __name__ == '__main__':
    gt_labels: np.ndarray = None
    gt_noisy_mask: np.ndarray = None
    cur_targets: np.ndarray = None
    main()
