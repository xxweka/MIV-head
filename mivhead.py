# Copyright (c) 2025 X. Xu; All Rights Reserved.

import datetime
from pathlib import Path
import random
import time
import numpy as np
import os
from copy import deepcopy

from torch import nn, optim
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode, Lambda, autoaugment
torchvision.disable_beta_transforms_warning()

import timm
from fsc_modeling import CAPBlock, Squeeze, CrossAttentionPooling, create_tasks, cross_l2normlize, TaskDataset, accuracy
from backbones.backbones import RNetBackbone, SDLRNet18Backbone, DINOBackbone, DeiT16Backbone, ClipBackbone, SWTBackbone, CNNBackbone
from config_args import args
from fvcore.nn import FlopCountAnalysis

timeout_second = 60 * 30
file_store = '/tmp/filestore'

# Component 1: Patch to Image
class P2I(nn.Module):
    def __init__(self, args, is_convnet=True):
        super().__init__()
        self.tau = args.tau
        self.convnet = is_convnet

        if is_convnet:
            self.embedproc0 = nn.ParameterList([])
            self.embedproc1 = nn.ParameterList([])
            for i, embedsize in enumerate(args.head_block_embed):
                p0 = torch.zeros((1, 1, embedsize))
                self.embedproc0.append(nn.Parameter(p0, requires_grad=True))
                if len(args.maxpool_sizes[i]) > 0:
                    p1 = torch.zeros((1, 1, embedsize))
                    self.embedproc1.append(nn.Parameter(p1, requires_grad=True))
        else:
            self.embedproc = nn.ModuleList([])
            self.embedproc0 = nn.ParameterList([])
            self.embedproc1 = nn.ParameterList([])
            for i in range(args.n_outputs):
                self.embedproc.append(
                    CrossAttentionPooling(
                        hid_dim=args.channel_size,
                        n_heads=1,
                        attn_method='sma',
                    )
                )
                if len(args.maxpool_sizes[i]) > 0:
                    p0 = torch.zeros((1, 1, args.channel_size))
                    self.embedproc0.append(nn.Parameter(p0, requires_grad=True))
                    p1 = torch.zeros((1, 1, args.channel_size))
                    self.embedproc1.append(nn.Parameter(p1, requires_grad=True))

    def forward(self, input_blocks):
        output_list = []
        for i, ylist in enumerate(input_blocks):
            pooled_output = []
            if self.convnet:
                # round 1
                batch, channel, _ = ylist[0].size()
                temp = np.sqrt(channel) / self.tau
                for y in ylist:
                    sq = y.size(-1)
                    if sq > 1:
                        if self.tau > 0:
                            pooled_output.append(self.attn_pool(y, self.embedproc0[i], temperature=temp))  # [(B,1,Ci)]
                        else:
                            pooled_output.append(y.mean(dim=-1, keepdim=True).mT)
                    else:
                        pooled_output.append(y.mT)
                # round 2
                _x = torch.cat(pooled_output, dim=1)  # (B,k,Ci)
                if _x.size(1) > 1:
                    if self.tau > 0:
                        output = self.attn_pool(_x.mT, self.embedproc1[i], temperature=temp)  # (B,1,Ci)
                    else:
                        output = _x.mean(dim=1, keepdim=True)
                else:
                    output = _x
            else:
                # round 1
                channel = ylist[0].size(-1)
                inv_temp = self.tau / np.sqrt(channel)
                for y in ylist:
                    batch, sq, _ = y.size()
                    if sq > 1:
                        _y, _ = self.embedproc[i](self.embedproc0[i], y, y, attnlogits_multiplier=inv_temp)
                        _x = F.layer_norm(_y, [channel])
                    else:
                        _x = y
                    pooled_output.append(_x)  # better than unnormalized
                # round 2
                _k = torch.cat(pooled_output, dim=1)
                if _k.size(1) > 1:
                    output, _ = self.embedproc[i](self.embedproc1[i], _k, _k, attnlogits_multiplier=inv_temp)
                else:
                    output = _k
            output_list.append(output.transpose(0, 1).contiguous())  # [(1,B,Ci)]

        return output_list

    def attn_pool(self, v, param_q, temperature):
        patches = F.normalize(v, dim=-2)  # (B,Ci,HW)
        attn_logits = param_q @ patches  # (1,1,Ci)@(B,Ci,HW)
        attn = F.softmax(attn_logits / temperature, dim=-1)  # (B,1,HW)
        return attn @ v.mT  # (B,1,Ci)


# Component 2: CAP for FSC based on query and prototype/bag representations
class MIVWrapper(nn.Module):
    def __init__(self, cap, args):
        super(MIVWrapper, self).__init__()
        self.head = cap

        if args.logits_param:
            self.logits_diag = nn.ParameterList([])
            for logits_param_size in args.head_block_embed:
                d = torch.empty((1, logits_param_size))
                nn.init.ones_(d)
                self.logits_diag.append(torch.nn.Parameter(d))
        else:
            curr_device = torch.cuda.current_device()
            self.logits_diag = [torch.ones((1, _size), device=curr_device) for _size in args.head_block_embed]
        self.l2logits = args.logits_l2type
        self.attnlogits_multipliers = args.attention_logits_rescale

        self.logits_ln = nn.ModuleList([])
        for chsz in args.head_block_embed:
            self.logits_ln.append(nn.LayerNorm(chsz))

        self.logits_l2norm = not args.post_layernorm
        if args.excite_param and args.excite_type.startswith('ve'):
            self.logits_excite = Squeeze(mode='variance')
        else:
            self.logits_excite = None

    def qb_logits(self, q, b, idx):
        if self.l2logits:
            logits = - (self.logits_diag[idx] * (q - b).square()).sum(dim=-1, keepdim=False)  # (1,C) * (q,L,C)
        else:
            logits = (self.logits_diag[idx] * q * b).sum(dim=-1, keepdim=False)
        return logits  # (q,L)

    def forward(self, querylayer, sslayer, mode='finetune'):
        logits_list = []
        unnorm_supplist = []
        qsize = querylayer[0].size(1)
        num_classes = len(sslayer[0])
        atts_per_block = []
        for i, support_set in enumerate(sslayer):
            query_set = querylayer[i]  # (l,q,Ci)
            channel_size = query_set.size(-1)
            layer = query_set.size(0)

            if channel_size * layer > 512:  # OOM
                atts_layer = [[] for _ in range(len(support_set))]
                sublogit = []
                for _l in range(layer):
                    qset = self.logits_ln[i](query_set[_l:_l + 1].transpose(0, 1))  # (q,1,Ci)
                    qvecs = []
                    bvecs = []
                    for j, support in enumerate(support_set):
                        sset = self.logits_ln[i](support[_l:_l + 1])  # (1,k,Ci)
                        qvec, bvec, att = self.head[i](
                            qset, sset,
                            excite_sqzed=None,
                            attnlogits_multiplier=self.attnlogits_multipliers[mode]
                        )  # (q,1,C), (q,1,C) [q, nhead, 1, nshot]
                        qvecs.append(qvec)
                        bvecs.append(bvec)
                        atts_layer[j].append(att)
                    support_bags = torch.cat(bvecs, dim=-2)  # (q,L,Ci)
                    unnorm_supplist.append(support_bags)
                    query = torch.cat(qvecs, dim=-2)  # (q,L,Ci)
                    if self.logits_l2norm:
                        query, support_bags = cross_l2normlize(query, support_bags)
                    sublogit.append(self.qb_logits(query, support_bags, idx=i))  # (q,L)
                logit = torch.stack(sublogit, dim=0)  # (l,q,L)
                atts = [torch.cat(al, dim=0) for al in atts_layer]  # [l*q, nhead, 1, nshot]
            else:
                atts = []
                query_set = query_set.view(-1, 1, channel_size)  # (l*q,1,Ci)
                query_set = self.logits_ln[i](query_set)

                qvecs = []
                bvecs = []
                for j, support in enumerate(support_set):
                    support = support.repeat_interleave(qsize, dim=0)  # (l*q,k,Ci)
                    support = self.logits_ln[i](support)

                    qvec, bvec, att = self.head[i](
                        query_set, support,
                        excite_sqzed=None,
                        attnlogits_multiplier=self.attnlogits_multipliers[mode]
                    )  # (l*q,1,C), (l*q,1,C) [l*q, nhead, 1, nshot]
                    qvecs.append(qvec)
                    bvecs.append(bvec)
                    atts.append(att)
                support_bags = torch.cat(bvecs, dim=-2)  # (l*q,L,Ci)
                unnorm_supplist.append(support_bags)
                query = torch.cat(qvecs, dim=-2)  # (l*q,L,Ci)
                if self.logits_l2norm:
                    query, support_bags = cross_l2normlize(query, support_bags)
                else:
                    query = F.layer_norm(query, [channel_size])
                    support_bags = F.layer_norm(support_bags, [channel_size])
                logit = self.qb_logits(query, support_bags, idx=i)  # (l*q,L)
                logit = logit.view(-1, qsize, num_classes)  # (l,q,L)

            atts_per_block.append(atts)
            logits_list.append(logit)
        ll = torch.cat(logits_list, dim=0)  # (b*l,q,L)
        logits = torch.logsumexp(ll, dim=0)  # (q,L)

        if mode.startswith('predict'):
            return logits, atts_per_block  # (q,L)
        return logits, unnorm_supplist  # [(l*q,L,Ci)]


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

    is_cnn = ('resnet' in args.backbone) or args.backbone.startswith('cnn')
    first_level = P2I(args, is_convnet=is_cnn).cuda(gpu_id)
    second_level = MIVWrapper(
        nn.ModuleList([
            CAPBlock(
                args=args,
                input_units=_size,
                output_units=_size,
                num_heads=args.num_heads[_n],
                head_init=args.head_init
            )
            for _n, _size in enumerate(args.head_block_embed)
        ]),
        args=args
    ).cuda(gpu_id)
    first_level.requires_grad_(True)
    second_level.requires_grad_(True)

    level1lr = args.finetune_lr * 0.05
    ft_param_groups = [
        dict(params=second_level.parameters(), lr=args.finetune_lr),
        dict(params=first_level.parameters(), lr=level1lr),
    ]
    ft_optimizer = optim.SGD(ft_param_groups, lr=args.finetune_lr, momentum=0.9,
                             weight_decay=args.weight_decay,
                             # nesterov=True
                             )
    if args.rank == 0 and debug:
        print(first_level, second_level, flush=True)
        fst_param = sum(p.numel() for p in first_level.parameters())
        snd_param = sum(p.numel() for p in second_level.parameters())
        print('Total # head params :', fst_param, snd_param, flush=True)
        fst_pn = 0
        snd_pn = 0
        fst_pm = 0
        snd_pm = 0
        for name, parm in first_level.named_parameters():
            if parm.requires_grad:
                fst_pn += 1
                fst_pm += parm.numel()
                print('trainable :', name, flush=True)
            else:
                print('non-trainable :', name, flush=True)
        for name, parm in second_level.named_parameters():
            if parm.requires_grad:
                snd_pn += 1
                snd_pm += parm.numel()
                print('trainable :', name, flush=True)
            else:
                print('non-trainable :', name, flush=True)
        print('Finetune # train params :', fst_pm, snd_pm, flush=True)
        print(f'Finetune variable # ({len(ft_param_groups)} groups): 1st-level={fst_pn}, 2nd-level params={snd_pn}', flush=True)

    eval_qset, (eval_sset, val_sset), eval_target = single_task
    support_sizes = [s.size(0) for s in eval_sset]
    num_shots = np.average(support_sizes)
    shot_factor = 1
    # shot_factor = 1.0 + 1.0 / num_shots

    # start timing when processing data
    adapt_start = time.time()
    eval_sset = [s.cuda(gpu_id, non_blocking=True) for s in eval_sset]
    raw_embedding_ss = infer_model(torch.cat(eval_sset, dim=0))  # blocks of allclass-tensor
    # embedding per class to avoid potential OOM
    val_sset = [s.cuda(gpu_id, non_blocking=True) for s in val_sset]
    raw_embedding_sq = [infer_model(sq) for sq in val_sset]   # classes of blocks
    pseudo_classes = support_classes(val_sset)

    # adaptation train
    with torch.autograd.set_detect_anomaly(True):
        first_level.train()
        second_level.train()
        ft_steps = args.finetune_steps

        ft_loss = []
        for step in range(ft_steps):
            ft_optimizer.zero_grad()

            embedding_ss = first_level(raw_embedding_ss)
            num_blks = len(embedding_ss)
            eval_supportset = [torch.split(allclass, support_sizes, dim=1) for allclass in embedding_ss]  # blocks of classes
            blocks = [[] for _ in range(num_blks)]  # blocks of classes
            for qry in raw_embedding_sq:
                for q, qlayer in enumerate(first_level(qry)):
                    blocks[q].append(qlayer)
            val_qset = [torch.cat(qb, dim=1) for qb in blocks]
            ft_logits, unnorms_blocks = second_level(val_qset, eval_supportset, mode='finetune')
            ftloss = finetune_loss(ft_logits / (args.temperature * shot_factor), pseudo_classes)

            ftloss.backward()
            ft_optimizer.step()
            ft_loss.append(ftloss.item())
    # training time
    time_task = time.time() - adapt_start
    if debug:
        print(f'Finetune {ft_steps} steps completed in {time_task}s per task', flush=True)
        print('Finetune loss =', ft_loss, '; shot-factor =', shot_factor, flush=True)
        print(f'Finetune {ft_steps} steps for generated {np.average([s.size(0) for s in val_sset])} shots (from {num_shots})', flush=True)

    # inference
    eval_qset = eval_qset.cuda(gpu_id, non_blocking=True)
    raw_embedding_qs = infer_model(eval_qset)
    eval_target = eval_target.cuda(gpu_id, non_blocking=True)
    first_level.eval()
    second_level.eval()
    with torch.no_grad():
        ss_embedding = first_level(raw_embedding_ss)
        eval_supportset = [torch.split(sp, support_sizes, dim=1) for sp in ss_embedding]
        eval_queryset = first_level(raw_embedding_qs)
        voutput, atts = second_level(eval_queryset, eval_supportset, mode='predict')
        voutput /= args.temperature
    # total latency time combining training and inference time
    time_latency = time.time() - adapt_start
    acc_time = accuracy(voutput, eval_target)
    acc_time.append(time_task)
    acc_time.append(time_latency)

    del infer_model, first_level, second_level, eval_qset, eval_sset, val_sset, val_qset, atts #, distloss
    del raw_embedding_ss, raw_embedding_qs, raw_embedding_sq, eval_supportset, eval_queryset, blocks
    torch.cuda.empty_cache()
    return acc_time, eval_target.size(0)


def count_flops(single_task, model, gpu_id, args, debug=False):
    infer_model = deepcopy(model)  # feature extraction
    infer_model.requires_grad_(False)
    infer_model.eval()

    is_cnn = ('resnet' in args.backbone) or args.backbone.startswith('cnn')
    first_level = P2I(args, is_convnet=is_cnn).cuda(gpu_id)
    second_level = MIVWrapper(
        nn.ModuleList([
            CAPBlock(
                args=args,
                input_units=_size,
                output_units=_size,
                num_heads=args.num_heads[_n],
                head_init=args.head_init
            )
            for _n, _size in enumerate(args.head_block_embed)
        ]),
        args=args
    ).cuda(gpu_id)
    first_level.requires_grad_(True)
    second_level.requires_grad_(True)
    ft_param_groups = [
        dict(params=second_level.parameters(), lr=args.finetune_lr),
        dict(params=first_level.parameters(), lr=args.finetune_lr * 0.05),
        # dict(params=distloss.parameters(), lr=5e-4)
    ]
    if args.rank == 0 and debug:
        print(first_level, second_level, flush=True)
        fst_param = sum(p.numel() for p in first_level.parameters())
        snd_param = sum(p.numel() for p in second_level.parameters())
        print('Finetune # params :', fst_param, snd_param, flush=True)
        fst_pn = 0
        snd_pn = 0
        fst_pm = 0
        snd_pm = 0
        for name, parm in first_level.named_parameters():
            if parm.requires_grad:
                fst_pn += 1
                fst_pm += parm.numel()
                print('trainable :', name, flush=True)
            else:
                print('non-trainable :', name, flush=True)
        for name, parm in second_level.named_parameters():
            if parm.requires_grad:
                snd_pn += 1
                snd_pm += parm.numel()
                print('trainable :', name, flush=True)
            else:
                print('non-trainable :', name, flush=True)
        print('Finetune # train params :', fst_pm, snd_pm, flush=True)
        print('Finetune # 1st-level, 2nd-level params :', len(ft_param_groups), fst_pn, snd_pn, flush=True)

    eval_qset, (eval_sset, val_sset), eval_target = single_task
    eval_sset = [s.cuda(gpu_id, non_blocking=True) for s in eval_sset]
    support_sizes = [s.size(0) for s in eval_sset]
    support_set = torch.cat(eval_sset, dim=0)
    raw_embedding_ss = infer_model(support_set)  # blocks of allclass-tensor
    # embedding per class to avoid potential OOM
    val_sset = [s.cuda(gpu_id, non_blocking=True) for s in val_sset]
    raw_embedding_sq = [infer_model(sq) for sq in val_sset]   # classes of blocks

    embedding_ss = first_level(raw_embedding_ss)
    num_blks = len(embedding_ss)
    eval_supportset = [torch.split(allclass, support_sizes, dim=1) for allclass in embedding_ss]  # blocks of classes
    blocks = [[] for _ in range(num_blks)]  # blocks of classes
    for qry in raw_embedding_sq:
        for q, qlayer in enumerate(first_level(qry)):
            blocks[q].append(qlayer)
    val_qset = [torch.cat(qb, dim=1) for qb in blocks]

    # calculate GFLOPs
    first_level.eval()
    second_level.eval()
    # with torch.no_grad():
    ssfl = FlopCountAnalysis(infer_model, support_set)
    # .tracer_warnings('none')
    ssfl.unsupported_ops_warnings(False)
    ssfl.uncalled_modules_warnings(False)
    sqflops = 0
    for sqset in val_sset:
        sqfl = FlopCountAnalysis(infer_model, sqset)
        sqfl.unsupported_ops_warnings(False)
        sqfl.uncalled_modules_warnings(False)
        sqflops += sqfl.total()
    bbflp = ssfl.total() + sqflops

    ssfl1 = FlopCountAnalysis(first_level, raw_embedding_ss)
    ssfl1.unsupported_ops_warnings(False)
    ssfl1.uncalled_modules_warnings(False)
    sqflops1 = 0
    for sqry in raw_embedding_sq:
        sqfl1 = FlopCountAnalysis(first_level, sqry)
        sqfl1.unsupported_ops_warnings(False)
        sqfl1.uncalled_modules_warnings(False)
        sqflops1 += sqfl1.total()
    flops1 = ssfl1.total() + sqflops1

    qsfl2 = FlopCountAnalysis(second_level, (val_qset, eval_supportset))
    qsfl2.unsupported_ops_warnings(False)
    qsfl2.uncalled_modules_warnings(False)
    flops2 = qsfl2.total()

    total_flops = (flops1 + flops2) * 3 * args.finetune_steps + bbflp
    gflops = total_flops / (2.0 ** 30)
    # print('Train FLOPS(GB) =', gflops, flush=True)

    return gflops


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

    if 'clip' in args.backbone:
        backbone = ClipBackbone(args)
    elif args.backbone.startswith(('vit', 'deit')):
        if 'dino' in args.backbone:
            backbone = DINOBackbone(args)
        else:
            backbone = DeiT16Backbone(args)
    elif args.backbone.startswith('sdl'):
        backbone = SDLRNet18Backbone(args)
    elif 'resnet' in args.backbone:
        backbone = RNetBackbone(args)
    elif args.backbone.startswith('swint'):
        backbone = SWTBackbone(args)
    elif args.backbone.startswith('cnn'):
        backbone = CNNBackbone(args, intermediate=True)
    else:
        raise NotImplementedError('Unknown backbone!')

    model = backbone.cuda(gpu)
    if args.rank == 0:
        print(model, flush=True)
    ft_criterion = nn.CrossEntropyLoss().cuda(gpu)
    print('No checkpoint, start from scratch', flush=True)

    # Data loading
    imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # constant_normalize = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
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
    if args.da_crop is None:
        da_resize = transforms.Resize(
            size=(args.resolution, args.resolution),
            interpolation=InterpolationMode.BICUBIC
        )
    else:
        da_resize = transforms.RandomResizedCrop(
            args.resolution,
            interpolation=InterpolationMode.BICUBIC,
            scale=(args.da_crop, 1.0)
        )
    distort_transform = transforms.Compose(
        [
            da_resize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandAugment(
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            imagenet_normalize
            # constant_normalize
        ]
    )

    kwargs = dict(
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valdir = Path(args.root)
    sampled_root = Path(args.sampled_path)
    if sampled_root.exists():
        print('Fixed samples exist and will be used', flush=True)
    else:
        sampled_root = None
    original_ds = args.num_da

    for ds in args.valdata[gpu]:
        args.num_da = original_ds
        fixed_tasks = create_tasks(data_root=valdir, sample_json=ds, sampled_root=sampled_root)
        # handle OOM caused by data-augmentation
        if args.shortlist is not None and ds.startswith(tuple(args.shortlist)):
            args.num_da = max(args.num_da - 5, 1)
            if isinstance(backbone, DINOBackbone) and args.patch_size == 8:
                if ds.startswith(('quickdraw', 'Food101')):
                    args.num_da = max(args.num_da - 5, 1)
                elif ds.startswith('cifar100'):
                    args.num_da = max(args.num_da - 8, 1)
            print('modified num_da for', ds, '=', args.num_da, flush=True)
        else:
            print('num_da of', ds, '=', args.num_da, flush=True)

        val_dataset = TaskDataset(
            args=args,
            tasks=fixed_tasks,
            transform=val_transform,
            aux_transform=ft_transform,
            distort_transform=distort_transform
        )
        val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)

        print(f'{ds} eval starts ..', flush=True)
        acc_round = []
        neval_round = []
        debug_print = True
        tot_adapt_time = 0
        tot_latency_time = 0
        gflops = 0
        for j, one_task in enumerate(val_loader):
            if args.gflops:
                gfl = count_flops(one_task, model, gpu, args, debug=debug_print)
                gflops += gfl
                if j % args.print_freq == 0:
                    print(f'{ds} {j}: ma_flops={gflops / (j+1)}GB', flush=True)
            else:
                if ft_criterion is not None:
                    try:
                        acc1, nevals = ft_infer(one_task, model, ft_criterion, gpu, args, debug=debug_print)
                    except torch.cuda.OutOfMemoryError as err:
                        print('OOM @ Rank =', args.rank, 'GPU/data =', gpu, ds, flush=True)
                        raise err
                else:
                    acc1, nevals = only_infer(one_task, model, gpu, temperature=args.temperature)
                # [acc, training-time, latency-time]
                top1acc = acc1[0].item()
                adtime = acc1[-2].item() if torch.is_tensor(acc1[-2]) else acc1[-2]
                ltime = acc1[-1].item() if torch.is_tensor(acc1[-1]) else acc1[-1]
                tot_adapt_time += adtime
                tot_latency_time += ltime
                acc_round.append(top1acc)
                neval_round.append(nevals)
                if j % args.print_freq == 0:
                    print(f'{ds} {j}: acc_num={int(top1acc * nevals / 100)}, tot_num={nevals}, '
                          f'acc={top1acc}, ma_acc={np.mean(acc_round)}, '
                          f'ma_adapt-time={tot_adapt_time / len(acc_round)}s, '
                          f'ma_latency-time={tot_latency_time / len(acc_round)}s', flush=True)
            if debug_print:
                debug_print = False

        tdn = ds.split('.json')[0]
        if args.gflops:
            print('#' * 90 + '\n' + tdn + ': avg. FLOPS(GB) per task =', gflops / len(val_loader), flush=True)
            print('#' * 90, flush=True)
        else:
            tot_tasks = len(acc_round)
            mean_acc = np.mean(acc_round)
            ci95_acc = 1.96 * np.std(acc_round) / np.sqrt(len(acc_round))
            if args.result_file is not None:
                result_path = Path('results/')
                if not result_path.exists():
                    result_path.mkdir()
                np.save('results/' + args.result_file + '_' + tdn + '.npy', np.asarray(acc_round))
                output_log = args.result_file + '_summary.log'
                with open('results/' + output_log, 'a') as log:
                    print('\n' + '#'*90 + '\n' + tdn + ':' + args.result_file, file=log)
                    print('tasks =', tot_tasks, '; avg. adaptation time per task =', tot_adapt_time / tot_tasks, file=log)
                    print(f'{ds} eval completed: mean_acc={mean_acc}, ci95={ci95_acc}', file=log)
                    print('#' * 90, file=log)
            else:
                print('\n' + '#'*90 + '\ntasks =', tot_tasks, '; avg. adaptation/latency time per task = (',
                      tot_adapt_time / tot_tasks, tot_latency_time / tot_tasks, ')', flush=True)
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
    args.da_crop = None

    args.shortlist = None
    if 'clip' in args.backbone:
        args.pretrained = 'openai/clip-vit-large-patch14'
        args.resolution = 224
        args.channel_size = 1024
        args.n_outputs = 2
        args.tau = 200
        args.maxpool_sizes = [[15, 11], [7]]  # CLS is already included
        args.num_da = 20
        args.shortlist = ['quickdraw', 'fungi', 'mscoco', 'traffic_sign', 'ChestX', 'cifar100', 'Food101']
        # args.patch_size = 14
        args.backbone += ('blk' + str(args.n_outputs))

    elif args.backbone.startswith('vit'):
        supervised = args.backbone.split('_')[-1]
        if supervised == 'dino':  # backbone: DINO
            pth_file = f'dino_{args.vit_arch}{args.patch_size}_pretrain.pth'
            args.pretrained = args.dinovit_path + '/' + pth_file
            # handle OOM, reducing da
            args.shortlist = ['quickdraw', 'fungi', 'ChestX', 'cifar100', 'mscoco', 'Food101']
        else:  # backbone: DeiT
            args.patch_size = 16
            if args.vit_arch.endswith('small'):
                args.pretrained = args.deit_path + '/deit_small_patch16_224-cd65a155.pth'
            else:
                args.pretrained = args.deit_path + '/deit_base_patch16_224-b5f2ef4d.pth'
        args.backbone = args.vit_arch
        args.resolution = 224
        args.channel_size = 768 if args.vit_arch.endswith('base') else 384
        args.n_outputs = 4
        args.dino_ckey = None  # no key in ckpt: 'student', 'teacher'
        args.tau = 200

        if args.patch_size == 8:
            args.maxpool_sizes = [[24, 20, 16], [12, 8], [], []]
            args.num_da = 10 if args.n_outputs > 4 else 15
        else:  # 16
            args.maxpool_sizes = [[13, 10], [7], [], []]
            args.num_da = 15 if args.n_outputs > 4 else 20
            if args.vit_arch.endswith('base'):
                args.n_outputs = 2
                args.maxpool_sizes = [[14, 10, 7], []]
                args.num_da = 30
                args.shortlist = ['quickdraw', ]
        args.backbone += (str(args.patch_size) + supervised + str(args.n_outputs) + 'blk')

    elif 'resnet' in args.backbone:
        # backbone: Resnet / SDL-Resnet
        args.resolution = 84 if args.backbone.endswith('18') else 224
        args.tau = 500
        args.rnet_modelname = args.backbone
        args.channel_size = 2048 if args.rnet_modelname.endswith('50') else 512
        if args.rnet_modelname.endswith('50'):
            args.num_da = 15
            # handle OOM, reducing da
            args.shortlist = ['quickdraw', 'ChestX', 'cifar100']
        else:
            args.num_da = 30

        if args.backbone.startswith('sdl'):
            args.rnet_pretrained = args.sdlrnet_path + '/model_best.pth.tar'
            args.rnet_lastn_blocks = 2
            args.maxpool_sizes = [[10, 9, 8, 7], [5, 4, 3]]  # sdlrn18 last2blocks: 11,6 already included
        elif args.backbone.startswith('dino'):
            args.rnet_pretrained = args.dinornet_path + '/dino_resnet50_pretrain.pth'
            args.rnet_lastn_blocks = 2
            args.maxpool_sizes = [[13, 11, 9, 8], [6, 5, 4]]  # 224: 56,28,14,7 are already included
        else:
            # args.torch_hub
            args.rnet_pretrained = 'DEFAULT'  # finetune, predict
            if args.rnet_modelname.endswith('18'):
                args.rnet_lastn_blocks = 3
                # rnet18 84: 21,11,6,3 are already included
                args.maxpool_sizes = [[10, 9, 8, 7], [5, 4], []]  # no need: [18, 15, 12],
            else:  # rnet34/50
                args.rnet_lastn_blocks = 2
                args.maxpool_sizes = [[13, 11, 9, 8], [6, 5, 4]]  # 224: 56,28,14,7 are already included
        args.backbone += ('blk' + str(args.rnet_lastn_blocks))

    elif args.backbone.startswith('swint'):
        args.pretrained = args.swin_path + '/simmim_finetune__swin_base__img224_window7__800ep'
        args.resolution = 224
        args.channel_size = 1024
        args.tau = 200
        args.n_outputs = 2
        args.maxpool_sizes = [[7], [4]]  # no CLS, 28,14,7,7
        args.num_da = 20
        args.shortlist = ['quickdraw', 'fungi', 'mscoco', 'traffic_sign', 'ChestX', 'cifar100', 'Food101']
        args.backbone += (str(args.n_outputs) + 'blk')

    else:
        assert args.backbone in ['cnn_regnet', 'cnn_densenet']
        args.resolution = 224
        args.channel_size = -1
        args.tau = 500
        args.rnet_modelname = args.backbone
        args.rnet_lastn_blocks = 2
        args.maxpool_sizes = [[13, 11, 9, 8], [6, 5, 4]]  # 112,56,28,14,7 are already included
        args.num_da = 20
        args.shortlist = ['ChestX', 'cifar100']
        if args.rnet_modelname.endswith('densenet'):
            args.rnet_pretrained = 'densenet161'
            args.maxpool_sizes = [[11, 9], [5]]  # 112,56,28,14,7 are already included
            args.num_da = 15
            args.shortlist.extend(['quickdraw', 'fungi'])
        elif args.rnet_modelname.endswith('regnet'):
            args.rnet_pretrained = 'regnety_016'
        else:
            raise NotImplementedError('This CNN backbone is unavailable!')

    args.excite_type = 'coe'
    args.logits_l2type = False  # NCC
    args.head_embed_size = args.channel_size
    args.head_output_size = args.head_embed_size
    if args.backbone.startswith(('vit', 'deit')):
        # DINO
        args.head_block_embed = [args.head_output_size] * args.n_outputs
        args.num_heads = [args.head_output_size // 64] * args.n_outputs  # 0
    elif 'resnet' in args.backbone:
        # RNet
        args.head_block_embed = [
            args.head_output_size // (2 ** (args.rnet_lastn_blocks - 1 - _x)) for _x in range(args.rnet_lastn_blocks)
        ]
        args.num_heads = [es // 64 for es in args.head_block_embed]
    elif args.backbone.startswith('swint'):  # swin-transformer
        dims = [args.head_output_size // 4, args.head_output_size // 2, args.head_output_size, args.head_output_size]
        nheads = [4, 8, 16, 32]
        args.head_block_embed = dims[(-args.n_outputs):]
        args.num_heads = nheads[(-args.n_outputs):]
    elif args.backbone.startswith('cnn'):
        if args.rnet_modelname.endswith('densenet'):
            dims = [96, 384, 768, 2112, 2208]
            nheads = [4, 8, 16, 32, 32]
        elif args.rnet_modelname.endswith('regnet'):
            dims = [32, 48, 120, 336, 888]
            nheads = [1, 2, 4, 8, 24]
        else:
            raise NotImplementedError('This CNN backbone is unavailable!')
        args.head_block_embed = dims[(-args.rnet_lastn_blocks):]
        args.num_heads = nheads[(-args.rnet_lastn_blocks):]
    else:
        raise NotImplementedError('This backbone is not implemented!')

    args.post_layernorm = False  # post_layernorm destroys multi-blocks
    if args.post_layernorm:
        args.temperature = np.sqrt(args.head_output_size)
    elif args.logits_l2type:
        args.temperature = 0.3
    else:
        args.temperature = 0.1
    attention_temperature = 0.1
    args.attention_logits_rescale = {
        'finetune': attention_temperature,
        'predict': attention_temperature
    }
    args.residual = True
    args.excite_param = True
    args.logits_param = False
    args.attention_method = 'dba_l1'  # 'sma'  #
    args.head_init = 'random'

    if args.gflops or not args.npresults:
        args.result_file = None
    else:
        args.result_file = 'miv' + str(args.resolution) + args.backbone
    print('GPU =', os.environ['CUDA_VISIBLE_DEVICES'], args, flush=True)
    torch.multiprocessing.spawn(main_worker, args=(args,), nprocs=args.world_size)


if __name__ == "__main__":
    main()
