# Copyright (c) authors. All Rights Reserved.

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
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
torchvision.disable_beta_transforms_warning()

from fsc_modeling import create_tasks, accuracy, PlainTaskDataset
from backbones import resnet18sdl as rnet18
from config_args import args
from fvcore.nn import FlopCountAnalysis

timeout_second = 60 * 30
file_store = '/tmp/filestore'


class pa(nn.Module):
    """
    pre-classifier alignment (PA) mapping from 'Universal Representation Learning from Multiple Domains for Few-shot Classification'
    (https://arxiv.org/pdf/2103.13841.pdf)
    """
    def __init__(self, feat_dim):
        super(pa, self).__init__()
        # define pre-classifier alignment mapping
        self.weight = nn.Parameter(torch.ones(feat_dim, feat_dim, 1, 1))
        self.weight.requires_grad = True

    def forward(self, x):
        if len(list(x.size())) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.conv2d(x, self.weight.to(x.device)).flatten(1)
        return x


class conv_tsa(nn.Module):
    def __init__(self, orig_conv, args):
        super(conv_tsa, self).__init__()
        # the original conv layer
        self.conv = deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        stride, _ = self.conv.stride
        # task-specific adapters
        if 'alpha' not in args.tsa_head_params:
            self.ad_type = 'none'
        else:
            self.ad_type = args.tsa_ad_type
            self.ad_form = args.tsa_ad_form
        if self.ad_type == 'residual':
            if self.ad_form == 'matrix' or planes != in_planes:
                self.alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
            else:
                self.alpha = nn.Parameter(torch.ones(1, planes, 1, 1))
        elif self.ad_type == 'serial':
            if self.ad_form == 'matrix':
                self.alpha = nn.Parameter(torch.ones(planes, planes, 1, 1))
            else:
                self.alpha = nn.Parameter(torch.ones(1, planes, 1, 1))
            self.alpha_bias = nn.Parameter(torch.ones(1, planes, 1, 1))
            self.alpha_bias.requires_grad = True
        if self.ad_type != 'none':
            self.alpha.requires_grad = True
        # print(self.ad_type, self.ad_form, self.alpha, flush=True)

    def forward(self, x):
        y = self.conv(x)
        if self.ad_type == 'residual':
            if self.alpha.size(0) > 1:
                # residual adaptation in matrix form
                y = y + F.conv2d(x, self.alpha, stride=self.conv.stride)
            else:
                # residual adaptation in channel-wise (vector)
                y = y + x * self.alpha
        elif self.ad_type == 'serial':
            if self.alpha.size(0) > 1:
                # serial adaptation in matrix form
                y = F.conv2d(y, self.alpha) + self.alpha_bias
            else:
                # serial adaptation in channel-wise (vector)
                y = y * self.alpha + self.alpha_bias
        return y


class TSA(nn.Module):
    """ Attaching task-specific adapters (alpha) and/or PA (beta) to the ResNet backbone """
    def __init__(self, args):
        super(TSA, self).__init__()

        if args.tsa_backbone.startswith('sdl'):
            orig_resnet = rnet18.resnet18(pretrained_model_path=args.rnet_pretrained, num_classes=None)
        elif args.tsa_backbone.startswith('dino'):
            orig_resnet = torchvision.models.resnet50(weights=None)
            orig_resnet.fc = nn.Identity()
            state_dict = torch.load(args.rnet_pretrained, map_location='cpu')
            orig_resnet.load_state_dict(state_dict, strict=False)
        else:
            os.environ['TORCH_HOME'] = args.torch_hub
            if args.tsa_backbone.endswith('18'):
                orig_resnet = torchvision.models.resnet18(weights=args.rnet_pretrained, zero_init_residual=True)
            elif args.tsa_backbone.endswith('34'):
                orig_resnet = torchvision.models.resnet34(weights=args.rnet_pretrained, zero_init_residual=True)
            else:
                orig_resnet = torchvision.models.resnet50(weights=args.rnet_pretrained, zero_init_residual=True)
            orig_resnet.fc = nn.Identity()

        self.args = args
        self.tsa_init = args.tsa_init
        # freeze the pretrained backbone
        for k, v in orig_resnet.named_parameters():
            v.requires_grad = False

        # attaching task-specific adapters (alpha) to each convolutional layers
        # note that we only attach adapters to residual blocks in the ResNet
        for block in orig_resnet.layer1:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m, args=args)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer2:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m, args=args)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer3:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m, args=args)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer4:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m, args=args)
                    setattr(block, name, new_conv)

        self.backbone = orig_resnet
        # attach pre-classifier alignment mapping (beta)
        if args.tsa_backbone.endswith('50'):
            feat_dim = orig_resnet.layer4[-1].bn3.num_features
        else:
            feat_dim = orig_resnet.layer4[-1].bn2.num_features
        beta = pa(feat_dim)
        setattr(self, 'beta', beta)

    def create_embed(self, context_images):
        tsa_opt = self.args.tsa_head_params
        if 'alpha' not in tsa_opt:
            with torch.no_grad():
                context_features = self.embed(context_images)
        else:
            # adapt features by task-specific adapters
            context_features = self.embed(context_images)
        if 'beta' in tsa_opt:
            # adapt feature by PA (beta)
            aligned_features = self.beta(context_features)
        else:
            aligned_features = context_features
        return aligned_features

    def forward(self, queries, suplist, mode='finetune'):
        self.backbone.eval()
        if mode.startswith('finetune'):
            qs = torch.cat(suplist, dim=0)
        else:
            assert queries is not None
            qs = queries

        support_set = [self.create_embed(spt).mean(dim=0) for spt in suplist]  # [(C)]
        support_set = torch.stack(support_set, dim=0).unsqueeze(0)  # (1,L,C)
        query_set = self.create_embed(qs).unsqueeze(1)  # (Q,1,C)
        logits = F.cosine_similarity(query_set, support_set, dim=-1, eps=1e-30)
        return logits

    def embed(self, x):
        if self.args.tsa_backbone.startswith('sdl'):
            return self.backbone.embed(x)
        # tv resnet
        return self.backbone(x)

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.backbone.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.backbone.named_parameters()]

    def reset(self):
        # initialize task-specific adapters (alpha)
        for k, v in self.backbone.named_parameters():
            if 'alpha' in k:
                # initialize each adapter as an identity matrix
                if self.tsa_init == 'eye':
                    if v.size(0) > 1:
                        v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)
                    else:
                        v.data = torch.ones(v.size()).to(v.device)
                    # for residual adapter, each adapter is initialized as identity matrix scaled by 0.0001
                    if self.args.tsa_ad_type == 'residual':
                        v.data = v.data * 0.0001
                    if 'bias' in k:
                        v.data = v.data * 0
                elif self.tsa_init == 'random':
                    # randomly initialization
                    v.data = torch.rand(v.data.size()).data.normal_(0, 0.001).to(v.device)
        # initialize pre-classifier alignment mapping (beta)
        v = self.beta.weight
        self.beta.weight.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)


def only_infer(single_task, infer_model, gpu_id, temperature=1.0):
    eval_qset, eval_sset, eval_target = single_task
    eval_qset = eval_qset.cuda(gpu_id, non_blocking=True)
    eval_sset = [vs.cuda(gpu_id, non_blocking=True) for vs in eval_sset]
    eval_target = eval_target.cuda(gpu_id, non_blocking=True)
    infer_model.eval()
    infer_model.requires_grad_(False)
    with torch.no_grad():
        voutput = infer_model(eval_qset, eval_sset, mode='predict')
        voutput /= temperature
        acc_time = accuracy(voutput, eval_target)
        acc_time.append(0)
        return acc_time, eval_target.size(0)


def ft_infer(single_task, infer_model, finetune_loss, gpu_id, args, debug=False):
    infer_model.reset()
    infer_model.eval()
    infer_model.requires_grad_(False)

    tsa_opt = args.tsa_head_params
    alpha_params = [v for k, v in infer_model.named_parameters() if 'alpha' in k]
    beta_params = [v for k, v in infer_model.named_parameters() if 'beta' in k]
    ft_param_groups = []
    ap = []
    bp = []
    if 'alpha' in tsa_opt:
        for a in alpha_params:
            a.requires_grad = True
            ap.append(a)
        ft_param_groups.append({'params': ap, 'lr': args.tsa_finetune_lr})
    if 'beta' in tsa_opt:
        for b in beta_params:
            b.requires_grad = True
            bp.append(b)
        ft_param_groups.append({'params': bp, 'lr': args.tsa_beta_lr})
    ft_optimizer = optim.Adadelta(ft_param_groups, lr=0.0)
    if args.rank == 0 and debug:
        print('Total # params :', sum(p.numel() for p in infer_model.parameters()), flush=True)
        print('Finetune # train params :', sum(p.numel() for p in infer_model.parameters() if p.requires_grad), flush=True)
        print(f'Finetune variable # ({len(ft_param_groups)} groups): alpha_params={len(ap)}, beta_params={len(bp)}', flush=True)

    eval_qset, val_sset, eval_target = single_task
    # start timing
    adapt_start = time.time()
    val_sset = [s.cuda(gpu_id, non_blocking=True) for s in val_sset]
    # finetune
    ft_steps = args.finetune_steps
    with torch.autograd.set_detect_anomaly(True):
        if args.amp:
            if args.amp_scale is None:
                scaler = torch.cuda.amp.GradScaler()
            else:
                scaler = torch.cuda.amp.GradScaler(init_scale=args.amp_scale)
                if debug:
                    print(f'Adjust AMP scale = {args.amp_scale}', flush=True)

        pseudo_classes = support_classes(val_sset)
        ft_loss = []
        for _ in range(ft_steps):
            ft_optimizer.zero_grad()
            infer_model.zero_grad()

            if args.amp:
                with torch.cuda.amp.autocast():
                    ft_logits = infer_model(None, val_sset, mode='finetune')
                    ftloss = finetune_loss(ft_logits / args.temperature, pseudo_classes)  # (K,L), (K)
                scaler.scale(ftloss).backward()
                scaler.step(ft_optimizer)
                scaler.update()
            else:
                ft_logits = infer_model(None, val_sset, mode='finetune')
                ftloss = finetune_loss(ft_logits / args.temperature, pseudo_classes)  # (K,L), (K)
                ftloss.backward()
                ft_optimizer.step()

            ft_loss.append(ftloss.item())
    time_task = time.time() - adapt_start
    if debug:
        print(f'Finetune {ft_steps} steps completed in {time_task}s per task', flush=True)
        print('Finetune loss =', ft_loss, flush=True)

    # inference
    eval_sset = val_sset
    eval_qset = eval_qset.cuda(gpu_id, non_blocking=True)
    eval_target = eval_target.cuda(gpu_id, non_blocking=True)
    infer_model.eval()
    infer_model.requires_grad_(False)

    with torch.no_grad():
        voutput = infer_model(eval_qset, eval_sset, mode='predict')
        voutput /= args.temperature

    time_latency = time.time() - adapt_start
    del infer_model, eval_qset, eval_sset, val_sset
    torch.cuda.empty_cache()
    acc_time = accuracy(voutput, eval_target)
    acc_time.append(time_task)
    acc_time.append(time_latency)
    return acc_time, eval_target.size(0)


def count_flops(single_task, infer_model, gpu_id, args, debug=False):
    infer_model.reset()
    infer_model.eval()
    infer_model.requires_grad_(False)

    tsa_opt = args.head_params
    alpha_params = [v for k, v in infer_model.named_parameters() if 'alpha' in k]
    beta_params = [v for k, v in infer_model.named_parameters() if 'beta' in k]
    ft_param_groups = []
    if 'alpha' in tsa_opt:
        ap = []
        for a in alpha_params:
            a.requires_grad = True
            ap.append(a)
        ft_param_groups.append({'params': ap, 'lr': args.finetune_lr})
    if 'beta' in tsa_opt:
        bp = []
        for b in beta_params:
            b.requires_grad = True
            bp.append(b)
        ft_param_groups.append({'params': bp, 'lr': args.beta_lr})
    if args.rank == 0 and debug:
        print('Finetune # params :', sum(p.numel() for p in infer_model.parameters()), flush=True)
        print('Finetune # train params :', sum(p.numel() for p in infer_model.parameters() if p.requires_grad), flush=True)
        print('Finetune # params :', len(ft_param_groups), flush=True)

    eval_qset, val_sset, eval_target = single_task
    val_sset = [s.cuda(gpu_id, non_blocking=True) for s in val_sset]

    # calculate GFLPOS
    infer_model.eval()
    with torch.no_grad():
        qsfl = FlopCountAnalysis(infer_model, (None, val_sset))
        qsfl.unsupported_ops_warnings(False)
        qsfl.uncalled_modules_warnings(False)
        total_flops = qsfl.total() * 3 * args.finetune_steps
    gflops = total_flops / (2.0 ** 30)
    # print('Train FLOPS(GB) =', gflops, flush=True)

    return gflops


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
    # print('Rank =', args.rank, 'GPU/data =', gpu, args.valdata[gpu], flush=True)

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

    if args.tsa_head_params:
        ft_criterion = nn.CrossEntropyLoss().cuda(gpu)
    else:
        ft_criterion = None
    # automatically resume from checkpoint if it exists
    print('No checkpoint, start from scratch', flush=True)

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
    for i, ds in enumerate(args.valdata[gpu]):
        # jumpto_next_ds = False
        args.amp = True  # handle OOM
        if ds.startswith('ISIC'):  # amp results in nan
            args.amp_scale = 2.0 ** 15
        else:
            args.amp_scale = None
        model = TSA(args).cuda(gpu)
        if args.rank == 0 and i == 0:
            print(model, flush=True)
        model.eval()
        fixed_tasks = create_tasks(data_root=valdir, sample_json=ds, sampled_root=sampled_root)
        val_dataset = PlainTaskDataset(
            fixed_tasks,
            transform=val_transform,
        )
        val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)

        print(f'{ds} eval starts ... Use amp? {args.amp}, init_scale = {args.amp_scale}', flush=True)
        acc_round = []
        neval_round = []
        debug_print = True
        tot_adapt_time = 0
        tot_latency_time = 0
        gflops = 0
        # assert ft_criterion is not None
        for j, one_task in enumerate(val_loader):
            args.amp_scale = None
            if args.gflops:
                gfl = count_flops(one_task, model, gpu, args, debug=debug_print)
                gflops += gfl
                if j % args.print_freq == 0:
                    print(f'{ds} {j}: ma_flops={gflops / (j+1)}GB', flush=True)
            else:
                try:
                    acc1, nevals = ft_infer(one_task, model, ft_criterion, gpu, args, debug=debug_print)
                except torch.cuda.OutOfMemoryError as err:
                    print('OOM @ Rank =', args.rank, 'GPU/data =', gpu, ds, err, flush=True)
                    raise err
                    # jumpto_next_ds = True
                    # break
                except RuntimeError as nan_err:
                    print('NaN @ GPU/data =', gpu, ds, nan_err, flush=True)
                    try:
                        args.amp_scale = 2.0 ** 12
                        acc1, nevals = ft_infer(one_task, model, ft_criterion, gpu, args, debug=True)
                    except Exception as e:
                        print(f'Still NaN: msg = {e}', flush=True)
                        raise e
                        # model.reset()
                        # acc1, nevals = only_infer(one_task, model, gpu, temperature=args.temperature)

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
        # if jumpto_next_ds:
        #     continue

        tdn = ds.split('.json')[0]
        if args.gflops:
            print('#' * 90 + '\n' + tdn + ': avg. FLOPS(GB) per task =', gflops / len(val_loader), flush=True)
            print('#' * 90, flush=True)
        else:
            tot_tasks = len(acc_round)
            nepi_tot = np.sum(neval_round)
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
                    print('tasks =', tot_tasks, '; avg. adaptation time per task =', tot_adapt_time / tot_tasks,
                          file=log)
                    print(f'{ds} eval completed: mean_acc={mean_acc}, ci95={ci95_acc}', file=log)
                    print('#' * 90, file=log)
            else:
                print('\n' + '#'*90 + '\ntasks =', tot_tasks, '; avg. adaptation/latency time per task = (',
                      tot_adapt_time / tot_tasks, tot_latency_time / tot_tasks, ')', flush=True)
                print(f'{ds} eval completed: mean_acc={mean_acc}, ci95={ci95_acc}, num_evals={nepi_tot}', flush=True)
                print('#'*90, flush=True)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    args.rank = 0
    args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
    args.world_size = torch.cuda.device_count()
    args.valdata = [[tds + '_' + args.test_type + '.json' for tds in args.test_data.split(' ')]]

    # # tsa
    # args.tsa_backbone = 'resnet18'  # 'sdl_resnet18'  # 'resnet50'  # 'dino_resnet50'  # 'resnet34'  #
    # args.tsa_ad_type = 'residual'
    # args.tsa_ad_form = 'matrix'
    # args.tsa_init = 'eye'
    # args.tsa_head_params = ['alpha', 'beta']
    # # test fine-tune
    # args.finetune_steps = 40
    # args.tsa_finetune_lr = 0.5
    # args.tsa_beta_lr = 1

    args.channel_size = 2048 if args.tsa_backbone.endswith('50') else 512
    args.temperature = 0.1
    args.resolution = 84 if args.tsa_backbone.endswith('18') else 224
    if args.tsa_backbone.startswith('sdl'):
        # Resnet SDL
        args.rnet_pretrained = args.sdlrnet_path + '/model_best.pth.tar'
    elif args.tsa_backbone.startswith('dino'):
        # DINO Resnet
        args.rnet_pretrained = args.dinornet_path + '/dino_resnet50_pretrain.pth'
    else:
        # # supervised Resnet off-the-shelf
        # args.torch_hub
        args.rnet_pretrained = 'DEFAULT'  # finetune, predict

    if args.gflops or not args.npresults:
        args.result_file = None
    else:
        args.result_file = 'tsa' + str(args.resolution) + args.tsa_backbone

    print('GPU =', os.environ['CUDA_VISIBLE_DEVICES'], args, flush=True)
    torch.multiprocessing.spawn(main_worker, args=(args,), nprocs=args.world_size)


if __name__ == "__main__":
    main()
