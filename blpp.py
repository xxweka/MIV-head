# Copyright (c) authors. All Rights Reserved.

import datetime
from pathlib import Path
import random
import sys
import time
import numpy as np
import multiprocessing as mp
from copy import deepcopy
import os

from torch import nn, optim
import torch
import torch.utils.data as data
from torch.nn.utils.weight_norm import WeightNorm
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode, Lambda, autoaugment
torchvision.disable_beta_transforms_warning()

from fsc_modeling import create_tasks, PlainTaskDataset, accuracy
from backbones.backbones import Backbone
from config_args import args

timeout_second = 60 * 30
file_store = '/tmp/filestore'


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm

        if outdim <= 200:
            self.scale_factor = 2  # a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github
        else:
            self.scale_factor = 10  #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        # print(x.shape, flush=True)
        x_norm = torch.norm(x, p=2, dim=-1).unsqueeze(-1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim=-1).unsqueeze(-1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor * cos_dist
        return scores


class BaselineFinetune(nn.Module):
    def __init__(self, args, feat_dim, n_way):
        super(BaselineFinetune, self).__init__()

        if args.classifier == 'bl':
            self.linear_clf = nn.Linear(feat_dim, n_way)
        elif args.classifier == 'blpp':
            self.linear_clf = distLinear(feat_dim, n_way)
        else:
            print("Invalid classifier option.")
            sys.exit()

    def forward(self, x):
        return self.linear_clf(x)


def adjust_learning_rate(args, optimizer, tasks_per_epoch, step):
    max_steps = args.epochs * tasks_per_epoch
    warmup_steps = args.lr_warmup * tasks_per_epoch
    # head
    q = 0.5 * (1 + np.cos(np.pi * step / max_steps))
    hlr = q + args.end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = hlr * args.lr_head
    # backbone
    if len(optimizer.param_groups) == 2:
        if step < (warmup_steps / 2):
            lr = 0.0
        elif step < warmup_steps:
            lr = 2 * step / warmup_steps - 1
        else:
            step -= warmup_steps
            max_steps -= warmup_steps
            q = 0.5 * (1 + np.cos(np.pi * step / max_steps))
            lr = q + args.end_lr * (1 - q)
        optimizer.param_groups[1]['lr'] = lr * args.lr_backbone


def only_infer(single_task, infer_model, gpu_id, temperature=1.0):
    # qset_tuple, sset_tuple, eval_target = single_task
    # eval_qset, _ = qset_tuple
    # eval_sset = [st[0] for st in sset_tuple]
    eval_qset, (eval_sset, _), eval_target = single_task
    eval_qset = eval_qset.cuda(gpu_id, non_blocking=True)
    eval_sset = [vs.cuda(gpu_id, non_blocking=True) for vs in eval_sset]
    eval_target = eval_target.cuda(gpu_id, non_blocking=True)

    infer_model.eval()
    infer_model.requires_grad_(False)
    with torch.no_grad():
        voutput, atts = infer_model(eval_qset, eval_sset, mode='predict')
        voutput /= temperature
        acc_time = accuracy(voutput, eval_target)
        acc_time.append(0)
        return acc_time, eval_target.size(0)


def ft_infer(single_task, model, finetune_loss, gpu_id, args, debug=False):
    infer_model = deepcopy(model)  # feature extraction
    infer_model.requires_grad_(False)
    infer_model.eval()

    # start timing when loading data
    adapt_start = time.time()
    eval_qset, eval_sset, eval_target = single_task
    eval_sset = [s.cuda(gpu_id, non_blocking=True) for s in eval_sset]
    train_qs = infer_model(torch.cat(eval_sset, dim=0))  # blocks of allclass-tensor
    # ss = [infer_model(sq) for sq in eval_sset]   # classes of blocks
    pseudo_classes = support_classes(eval_sset)

    num_class = len(eval_sset)
    linear_clf = BaselineFinetune(args, feat_dim=args.channel_size, n_way=num_class).cuda(gpu_id)
    linear_clf.requires_grad_(True)
    ft_optimizer = torch.optim.Adam(linear_clf.parameters(), lr=args.bl_finetune_lr)

    if args.rank == 0 and debug:
        print(linear_clf, flush=True)
        fst_param = sum(p.numel() for p in linear_clf.parameters())
        print('Finetune # params :', fst_param, flush=True)

    # adaptation train
    with torch.autograd.set_detect_anomaly(True):
        linear_clf.train()
        ft_steps = args.finetune_steps
        ft_loss = []
        for step in range(ft_steps):
            ft_optimizer.zero_grad()
            scores = linear_clf(train_qs)
            ftloss = finetune_loss(scores, pseudo_classes)
            ftloss.backward()
            ft_optimizer.step()
            ft_loss.append(ftloss.item())
    time_task = time.time() - adapt_start
    if debug:
        print(f'Finetune {ft_steps} steps completed in {time_task}s per task', flush=True)
        print('Finetune loss =', ft_loss, flush=True)

    # inference
    eval_qset = eval_qset.cuda(gpu_id, non_blocking=True)
    test_qs = infer_model(eval_qset)
    eval_target = eval_target.cuda(gpu_id, non_blocking=True)
    linear_clf.eval()
    with torch.no_grad():
        voutput = linear_clf(test_qs)
    acc_time = accuracy(voutput, eval_target)
    acc_time.append(time_task)

    del infer_model, linear_clf, eval_qset, eval_sset, train_qs, test_qs, scores, ftloss, ft_loss #, ss
    torch.cuda.empty_cache()
    return acc_time, eval_target.size(0)


def sup2qry(suplist):  # [(l,k,C)]
    pseudo_query = torch.cat(suplist, dim=1)  # (l,K,C)
    # mask
    qsize = pseudo_query.size(1)
    cum_ssize = 0
    mask = []
    for support in suplist:
        ssize = support.size(1)
        maskmap = torch.ones(qsize, ssize).to(device=pseudo_query.device, dtype=torch.bool)
        if ssize > 1:
            maskmap[range(cum_ssize, cum_ssize+ssize), range(ssize)] = False
        mask.append(maskmap)
        cum_ssize += ssize
    return pseudo_query, mask


def support_classes(suplist):
    class_list = []
    sdevice = None
    for i, support in enumerate(suplist):
        if sdevice is None:
            sdevice = support.device
        ssize = support.size(0)
        class_list.extend([i] * ssize)
    return torch.tensor(class_list, device=sdevice)


def main_worker(gpu, args):
    args.rank += gpu
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    print('Rank =', args.rank, 'GPU/data =', gpu, args.valdata[gpu], flush=True)

    # replace the following with your own setting of GPU-server
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )
    # torch.distributed.init_process_group(
    #     backend="nccl",
    #     store=torch.distributed.FileStore(file_store, args.world_size),
    #     world_size=args.world_size,
    #     rank=args.rank,
    #     timeout=datetime.timedelta(seconds=timeout_second)
    # )

    model = Backbone(args).cuda(gpu)
    if args.rank == 0:
        print(model, flush=True)
    ft_criterion = nn.CrossEntropyLoss().cuda(gpu)
    print('No checkpoint, start from scratch', flush=True)

    # Data loading
    imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    constant_normalize = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    val_transform = transforms.Compose(
        [
            transforms.Resize(
                size=(args.resolution, args.resolution),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            imagenet_normalize
            # constant_normalize
        ]
    )
    ft_transform = None
    distort_transform = None

    kwargs = dict(
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valdir = Path(args.root)
    sampled_root = valdir.parent.joinpath('sampled/')
    if sampled_root.exists():
        print('Fixed samples exist and will be used', flush=True)

    for ds in args.valdata[gpu]:
        fixed_tasks = create_tasks(data_root=valdir, sample_json=ds, sampled_root=sampled_root)
        val_dataset = PlainTaskDataset(
            tasks=fixed_tasks,
            transform=val_transform,
            aux_transform=ft_transform
        )
        val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)

        print(f'{ds} eval starts ..', flush=True)
        acc_round = []
        neval_round = []
        debug_print = True
        tot_adapt_time = 0
        for j, one_task in enumerate(val_loader):
            if ft_criterion is not None:
                try:
                    acc1, nevals = ft_infer(one_task, model, ft_criterion, gpu, args, debug=debug_print)
                except torch.cuda.OutOfMemoryError as err:
                    print('OOM @ Rank =', args.rank, 'GPU/data =', gpu, ds, flush=True)
                    raise err
            else:
                acc1, nevals = only_infer(one_task, model, gpu, temperature=args.temperature)
            top1acc = acc1[0].item()
            adtime = acc1[-1].item() if torch.is_tensor(acc1[-1]) else acc1[-1]
            tot_adapt_time += adtime
            acc_round.append(top1acc)
            neval_round.append(nevals)

            if j % args.print_freq == 0:
                print(f'{ds} {j}: acc_num={int(top1acc * nevals / 100)}, tot_num={nevals}, '
                      f'acc={top1acc}, ma_acc={np.mean(acc_round)}, ma_adapt-time={tot_adapt_time / len(acc_round)}s', flush=True)
            if debug_print:
                debug_print = False

        tot_tasks = len(acc_round)
        mean_acc = np.mean(acc_round)
        ci95_acc = 1.96 * np.std(acc_round) / np.sqrt(len(acc_round))
        if args.result_file is not None:
            tdn = ds.split('.json')[0]
            np.save('results/' + args.result_file + '_' + tdn + '.npy', np.asarray(acc_round))
            output_log = args.result_file + '_summary.log'
            with open('results/' + output_log, 'a') as log:
                print('\n' + '#'*90 + '\n' + tdn + ':' + args.result_file, file=log)
                print('tasks =', tot_tasks, '; avg. adaptation time per task =', tot_adapt_time / tot_tasks,
                      file=log)
                print(f'{ds} eval completed: mean_acc={mean_acc}, ci95={ci95_acc}', file=log)
                print('#' * 90, file=log)
        else:
            print('\n' + '#'*90 + '\ntasks =', tot_tasks, '; avg. adaptation time per task =', tot_adapt_time / tot_tasks, flush=True)
            print(f'{ds} eval completed: mean_acc={mean_acc}, ci95={ci95_acc}', flush=True)
            print('#'*90, flush=True)


def main():
    if ('clip' in args.backbone) or ('cnn' in args.backbone):
        os.environ['HUGGINGFACE_HUB_CACHE'] = args.hf_hub
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    args.rank = 0
    args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
    args.world_size = torch.cuda.device_count()
    args.valdata = [[tds + '_' + args.test_type + '.json' for tds in args.test_data.split(' ')]]
    # # placeholder
    args.n_outputs = None
    args.rnet_lastn_blocks = None
    args.maxpool_sizes = None

    if 'clip' in args.backbone:
        args.pretrained = 'openai/clip-vit-large-patch14'
        args.resolution = 224
        args.channel_size = 1024

    elif args.backbone.startswith('vit'):
        supervised = args.backbone.split('_')[-1]
        # backbone: DINO
        if supervised == 'dino':
            pth_file = f'dino_{args.vit_arch}{args.patch_size}_pretrain.pth'
            args.pretrained = args.dinovit_path + '/' + pth_file
        else:
            args.patch_size = 16
            if args.vit_arch.endswith('base'):
                args.pretrained = args.deit_path + '/deit_base_patch16_224-b5f2ef4d.pth'
            else:
                args.pretrained = args.deit_path + '/deit_small_patch16_224-cd65a155.pth'
        args.backbone = args.vit_arch
        args.resolution = 224
        args.channel_size = 768 if args.vit_arch.endswith('base') else 384
        args.backbone += (str(args.patch_size) + supervised)

    elif 'resnet' in args.backbone:
        # backbone: Resnet / SDL-Resnet
        args.resolution = 84 if args.backbone.endswith('18') else 224
        args.rnet_modelname = args.backbone
        args.channel_size = 2048 if args.rnet_modelname.endswith('50') else 512

        if args.backbone.startswith('sdl'):
            args.rnet_pretrained = args.sdlrnet_path + '/model_best.pth.tar'
        elif args.backbone.startswith('dino'):
            args.rnet_pretrained = args.dinornet_path + '/dino_resnet50_pretrain.pth'
        else:
            # args.torch_hub
            args.rnet_pretrained = 'DEFAULT'

    elif args.backbone.startswith('swint'):
        args.pretrained = args.swin_path + '/simmim_finetune__swin_base__img224_window7__800ep'
        args.resolution = 224
        args.channel_size = 1024

    elif args.backbone.startswith('cnn'):
        args.resolution = 224
        args.rnet_modelname = args.backbone
        if args.rnet_modelname.endswith('densenet'):
            args.rnet_pretrained = 'densenet161'
            args.channel_size = 2208
        elif args.rnet_modelname.endswith('regnet'):
            args.rnet_pretrained = 'regnety_016'
            args.channel_size = 888
        else:
            raise NotImplementedError('This CNN backbone is unavailable!')

    else:
        raise NotImplementedError('This backbone is not implemented!')

    # meta-test
    args.finetune_steps = 400
    if args.npresults:
        args.result_file = args.classifier + str(args.resolution) + args.backbone
    else:
        args.result_file = None
    print('GPU =', os.environ['CUDA_VISIBLE_DEVICES'], args, flush=True)
    torch.multiprocessing.spawn(main_worker, args=(args,), nprocs=args.world_size)


if __name__ == "__main__":
    main()
