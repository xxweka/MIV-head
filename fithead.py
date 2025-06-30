# Copyright (c) authors. All Rights Reserved.

from pathlib import Path
import random
import sys
import time
import datetime
import numpy as np
import os
from copy import deepcopy
from functools import partial

from torch import nn, optim
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.distributions import multivariate_normal, constraints
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
torchvision.disable_beta_transforms_warning()

import timm
# from timm.models.vision_transformer import VisionTransformer, _cfg
from fsc_modeling import create_tasks, PlainTaskDataset, accuracy
from backbones import DINO_ViT as dino, resnet18sdl as rnet18
from config_args import args

timeout_second = 60 * 30
file_store = '/tmp/filestore'


class Backbone(nn.Module):
    def __init__(self, args):
        super(Backbone, self).__init__()
        self.special_embed = False
        if args.backbone.startswith(('vit', 'deit')):
            if 'dino' in args.backbone:
                if args.vit_arch.startswith('deit'):
                    self.nnmodel = dino.vit_small(args.patch_size)
                else:
                    self.nnmodel = dino.vit_base(args.patch_size)
                dino.load_pretrained_weights(
                    model=self.nnmodel,
                    pretrained_weights=args.pretrained,
                    patch_size=args.patch_size,
                    model_name=None,
                    checkpoint_key=args.dino_ckey
                )
            else:
                assert args.patch_size == 16
                nheads = args.channel_size // 64
                self.nnmodel = timm.models.vision_transformer.VisionTransformer(
                    patch_size=args.patch_size, embed_dim=args.channel_size, depth=12, num_heads=nheads, mlp_ratio=4,
                    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0
                )
                self.nnmodel.default_cfg = timm.models.vision_transformer._cfg()
                self.nnmodel.head = nn.Identity()
                checkpoint = torch.load(args.pretrained, map_location='cpu')
                msg = self.nnmodel.load_state_dict(checkpoint['model'], strict=False)
                print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained, msg))
        elif args.backbone.startswith('sdl'):
            self.nnmodel = rnet18.resnet18(pretrained_model_path=args.rnet_pretrained, num_classes=None)
            self.special_embed = True
        else:
            if args.rnet_modelname.startswith('dino'):
                self.nnmodel = torchvision.models.resnet50(weights=None)
                state_dict = torch.load(args.rnet_pretrained, map_location='cpu')
                self.nnmodel.load_state_dict(state_dict, strict=False)
            else:
                # torch.hub.set_dir(args.torch_hub)
                os.environ['TORCH_HOME'] = args.torch_hub
                if args.rnet_modelname.endswith('18'):
                    self.nnmodel = torchvision.models.resnet18(weights=args.rnet_pretrained, zero_init_residual=True)
                elif args.rnet_modelname.endswith('34'):
                    self.nnmodel = torchvision.models.resnet34(weights=args.rnet_pretrained, zero_init_residual=True)
                else:
                    self.nnmodel = torchvision.models.resnet50(weights=args.rnet_pretrained, zero_init_residual=True)
            self.nnmodel.fc = nn.Identity()

    def forward(self, input):
        self.nnmodel.eval()
        if self.special_embed:
            output = self.nnmodel.embed(input)  # (Q,C)
        else:
            output = self.nnmodel(input)
        return output


class NaiveBayesPredictor(nn.Module):
    def __init__(self, args, eps=1e-5):
        super(NaiveBayesPredictor, self).__init__()
        self.args = args
        if self.args.fithead_classifier == "qda":
            self.class_cov_weight = nn.Parameter(torch.tensor([0.5]))
            self.task_cov_weight = nn.Parameter(torch.tensor([0.5]))
            self.cov_reg_weight = nn.Parameter(torch.tensor([1.0]))
        elif self.args.fithead_classifier == "lda":
            self.task_cov_weight = nn.Parameter(torch.tensor([1.0]))
            self.cov_reg_weight = nn.Parameter(torch.tensor([1.0]))
        self.eps = eps
        return

    def predict_naive_bayes(self, target_features, class_means, class_trils):
        class_distributions = multivariate_normal.MultivariateNormal(
            loc=class_means,
            scale_tril=class_trils
        )
        return class_distributions.log_prob(target_features.unsqueeze(dim=1))  # (num_targets, num_classes)

    def predict_protonets(self, target_features, class_means, class_trils):
        num_target_features = target_features.shape[0]
        num_prototypes = class_means.shape[0]
        distances = (target_features.unsqueeze(1).expand(num_target_features, num_prototypes, -1) -
                    class_means.unsqueeze(0).expand(num_target_features, num_prototypes, -1)).pow(2).sum(dim=2)
        return -distances

    def compute_class_means_and_trils(self, features):  # list
        means = []
        trils = []
        all_classes = torch.cat(features, dim=0)
        sz = all_classes.size(1)
        # need task cov for qda and lda, but not protonets
        if self.args.fithead_classifier != "protonets":
            task_covariance_estimate = self._estimate_cov(all_classes)
        # lda just needs the task cov
        if self.args.fithead_classifier == "lda":
            eye = torch.eye(sz, sz).to(device=all_classes.device)
            cov_reg_weight = self.cov_reg_weight if self.cov_reg_weight > 0 else F.relu(self.cov_reg_weight) + self.eps
            cov_est = F.relu(self.task_cov_weight) * task_covariance_estimate + cov_reg_weight * eye
            try:
                std = self._lower_triangular(cov_est)
                assert constraints.lower_cholesky.check(std), 'covariance NOT positive definite'
            except Exception as e:
                print(std, self.task_cov_weight, self.cov_reg_weight, cov_reg_weight)
                print(e, cov_est, flush=True)
                eps = self.eps * 100
                try:
                    std = self._lower_triangular(cov_est + eps * eye)
                    assert constraints.lower_cholesky.check(std), 'covariance still NOT positive definite'
                except:
                    ind_cov = (F.relu(task_covariance_estimate) + eps) * eye
                    std = torch.sqrt(F.relu(self.task_cov_weight) * ind_cov + F.relu(self.cov_reg_weight) * eye)
                    assert not std.isnan().any()
            trils.append(std)
        for class_features in features:
            # mean pooling examples to form class means
            means.append(self._mean_pooling(class_features).squeeze())
            # compute class cov for qda
            if self.args.fithead_classifier == "qda":
                trils.append(self._lower_triangular(
                    F.relu(self.class_cov_weight) * self._estimate_cov(class_features) +
                    F.relu(self.task_cov_weight) * task_covariance_estimate +
                    F.relu(self.cov_reg_weight) * torch.eye(sz, sz).to(device=class_features.device))
                )
        means = torch.stack(means)
        if self.args.fithead_classifier != "protonets":
            trils = torch.stack(trils)
        return means, trils

    @staticmethod
    def _estimate_cov(examples):
        if examples.size(0) > 1:
            return torch.cov(examples.t(), correction=1)
        else:
            factor = 1.0 / (examples.size(1) - 1)
            examples = examples - torch.mean(examples, dim=1, keepdim=True)
            return factor * examples.matmul(examples.t()).squeeze()

    @staticmethod
    def _lower_triangular(matrix):
        return torch.linalg.cholesky(matrix)

    @staticmethod
    def _mean_pooling(x):
        return torch.mean(x, dim=0, keepdim=True)


class FITHead(nn.Module):
    def __init__(self, args):
        super(FITHead, self).__init__()
        self.predictor = NaiveBayesPredictor(args)
        self.means = None
        self.trils = None
        self.all_means = None
        if (args.fithead_classifier == "lda") or (args.fithead_classifier == "qda"):
            self.predict_fn = self.predictor.predict_naive_bayes
        elif args.fithead_classifier == "protonets":
            self.predict_fn = self.predictor.predict_protonets
        else:
            print("Invalid classifier option.")
            sys.exit()

    def configure_from_features(self, context_features):
        self.means, self.trils = self.predictor.compute_class_means_and_trils(context_features)

    def predict(self, target_features):
        return self.predict_fn(target_features, self.means, self.trils)
    #
    # def set_classifier_subset(self, subset_idxs):
    #     if self.all_means is None:
    #         self.all_means = self.means.clone()
    #     self.means = self.all_means[subset_idxs]
    #
    # def set_all_classes_classifier(self):
    #     self.means = self.all_means.clone()
    #     self.all_means = None


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

    fh = FITHead(args).cuda(gpu_id)
    fh.requires_grad_(True)
    ft_param_groups = [
        dict(params=fh.parameters(), lr=args.fithead_finetune_lr)
    ]
    ft_optimizer = torch.optim.Adam(ft_param_groups, lr=args.fithead_finetune_lr)
    if args.rank == 0 and debug:
        print(fh, flush=True)
        fst_param = sum(p.numel() for p in fh.parameters())
        print('Total # head params :', fst_param, flush=True)
        fst_pn = 0
        fst_pm = 0
        for name, parm in fh.named_parameters():
            if parm.requires_grad:
                fst_pn += 1
                fst_pm += parm.numel()
                print('trainable :', name, flush=True)
            else:
                print('non-trainable :', name, flush=True)
        print('Finetune # train params :', fst_pm, flush=True)
        print(f'Finetune variable # ({len(ft_param_groups)} groups): {fst_pn}', flush=True)

    # no need to start timing
    adapt_start = time.time()
    eval_qset, eval_sset, eval_target = single_task
    eval_sset = [s.cuda(gpu_id, non_blocking=True) for s in eval_sset]
    train_qs = infer_model(torch.cat(eval_sset, dim=0))  # blocks of allclass-tensor
    ss = [infer_model(sq) for sq in eval_sset]   # classes of blocks
    pseudo_classes = support_classes(eval_sset)

    # adaptation train
    with torch.autograd.set_detect_anomaly(True):
        fh.train()
        ft_steps = args.finetune_steps
        ft_loss = []
        for step in range(ft_steps):
            fh.configure_from_features(ss)
            ft_optimizer.zero_grad()

            logits = fh.predict(train_qs)
            ftloss = finetune_loss(logits, pseudo_classes)
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
    fh.eval()
    with torch.no_grad():
        voutput = fh.predict(test_qs)
    acc_time = accuracy(voutput, eval_target)
    acc_time.append(time_task)

    del infer_model, fh, eval_qset, eval_sset, train_qs, test_qs, ss, logits, ftloss, ft_loss
    torch.cuda.empty_cache()
    return acc_time, eval_target.size(0)


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
    # automatically resume from checkpoint if it exists
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
    for ds in args.valdata[gpu]:
        fixed_tasks = create_tasks(data_root=valdir, sample_json=ds, sampled_root=sampled_root)
        val_dataset = PlainTaskDataset(
            tasks=fixed_tasks,
            transform=val_transform
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
            result_path = Path('results/')
            if not result_path.exists():
                result_path.mkdir()
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
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    args.rank = 0
    args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
    args.world_size = torch.cuda.device_count()
    args.valdata = [[tds + '_' + args.test_type + '.json' for tds in args.test_data.split(' ')]]

    # FiT head
    args.finetune_steps = 400
    if args.backbone.startswith('vit'):
        supervised = args.backbone.split('_')[-1]
        if supervised == 'dino':
            pth_file = f'dino_{args.vit_arch}{args.patch_size}_pretrain.pth'
            args.pretrained = args.dinovit_path + '/' + pth_file
        else:
            args.patch_size = 16
            if args.vit_arch.endswith('small'):
                args.pretrained = args.deit_path + '/deit_small_patch16_224-cd65a155.pth'
            else:
                args.pretrained = args.deit_path + '/deit_base_patch16_224-b5f2ef4d.pth'
        args.backbone = args.vit_arch
        args.resolution = 224
        args.channel_size = 768 if args.vit_arch.endswith('base') else 384
        args.dino_ckey = None  # no key in ckpt: 'student', 'teacher'
        args.backbone += (str(args.patch_size) + supervised)
    else:
        # backbone: Resnet / SDL-Resnet
        args.resolution = 84 if args.backbone.endswith('18') else 224
        args.tau = 500
        args.rnet_modelname = args.backbone
        args.channel_size = 2048 if args.rnet_modelname.endswith('50') else 512
        if args.backbone.startswith('sdl'):
            args.rnet_pretrained = args.sdlrnet_path + '/model_best.pth.tar'
        elif args.backbone.startswith('dino'):
            args.rnet_pretrained = args.dinornet_path + '/dino_resnet50_pretrain.pth'
        else:
            # args.torch_hub
            args.rnet_pretrained = 'DEFAULT'  # finetune, predict  # None  # pretrain

    if args.npresults:
        args.result_file = 'fithead' + str(args.resolution) + args.backbone
    else:
        args.result_file = None
    print('GPU =', os.environ['CUDA_VISIBLE_DEVICES'], args, flush=True)
    torch.multiprocessing.spawn(main_worker, args=(args,), nprocs=args.world_size)


if __name__ == "__main__":
    main()
