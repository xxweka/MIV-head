# Copyright (c) 2025 X. Xu; All Rights Reserved.

import os
import numpy as np
import json
from collections import defaultdict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from PIL import Image, ImageOps, ImageFilter, PngImagePlugin, ImageFile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch import Tensor


def check_npy(path: str) -> bool:
    return path.lower().endswith('.npy')


def pil_loader(path: str) -> Image.Image:
    if check_npy(path):
        array = np.load(path)
        img = Image.fromarray(array)
        return img.convert('RGB')

    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        try:
            img = Image.open(f)
        except ValueError as VE:
            if str(VE) == 'Decompressed Data Too Large':
                print('found a decompression bomb at ' + path, flush=True)
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 ** 2)
                img = Image.open(f)
                pass
        return img.convert('RGB')


def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


def create_tasks(data_root, sample_json, sampled_root=None, test_type='standard', num_tasks=-1):
    class_splits = defaultdict(list)
    ds_name = sample_json.replace('_' + test_type + '.json', '').strip()
    curr_path = None
    if sampled_root is not None:
        curr_path = sampled_root.joinpath(sample_json)

    if curr_path is not None and curr_path.exists():
        with open(curr_path, 'r') as sampled:
            tasks = json.load(sampled)
    else:
        print(ds_name, 'init starts', flush=True)
        root = data_root.joinpath(ds_name)
        classes = os.listdir(root)
        count_total = 0
        for c in classes:
            parent = root.joinpath(c)
            files = os.listdir(parent)
            class_splits[c] = [parent.joinpath(f).absolute().as_posix() for f in files]
            count_total += len(files)
        print(ds_name, 'init completed', len(class_splits), count_total, flush=True)
        tasks = []
        with open(data_root.joinpath(sample_json), 'r') as jf:
            samples = json.load(jf)
        if num_tasks < 0:
            num_tasks = len(samples)
        for i, task_id in enumerate(samples):
            if i >= num_tasks:
                break
            asample = samples[task_id]
            atask = {}
            atask['query'] = []
            atask['qcidx'] = []
            atask['support'] = []
            for c in asample:
                cls = asample[c]['class']
                qnum = asample[c]['query']
                ninst = qnum + asample[c]['support']

                # two equivalent ways of sampling
                chosen_insts = class_splits[cls]
                np.random.shuffle(chosen_insts)
                # inst_list = class_splits[cls]
                # chosen_insts = np.random.choice(inst_list, ninst, replace=False).tolist()

                atask['query'].extend(chosen_insts[:qnum])
                atask['qcidx'].extend([int(c)] * qnum)
                atask['support'].append(chosen_insts[qnum:ninst])
                assert int(c) == len(atask['support']) - 1
            tasks.append(atask)
    return tasks


def create_fixed_episodes(schema_root, sampled_path, sample_schemas):
    root_folder = Path(schema_root)
    output_folder = Path(sampled_path)
    # assert output_folder.exists(), f'Output folder {output_folder} does not exist!'
    if not output_folder.exists():
        output_folder.mkdir()

    for sample_json in sample_schemas:
        output_file = output_folder.joinpath(sample_json)
        if not output_file.exists():
            fixed_tasks = create_tasks(data_root=root_folder, sample_json=sample_json)
            with open(output_file, 'w') as jf:
                json.dump(fixed_tasks, jf)
            print(f'Output fixed episodes: {output_file}', flush=True)


def cross_l2normlize(a, b):
    centralize = b.mean(dim=-2, keepdim=True)
    return F.normalize(a - centralize, dim=-1), F.normalize(b - centralize, dim=-1)


def _masked_moments(values, mask_identity, axis, keep_dims=False):
    assert len(mask_identity.size()) == len(values.size())
    sumofweighting = mask_identity.sum(dim=axis, keepdim=True)  #.nan_to_num(posinf=0., neginf=0., nan=0.)

    masked_values = values * mask_identity
    mean = masked_values.sum(dim=axis, keepdim=True) / sumofweighting
    masked_sq = (masked_values - mean).square() * mask_identity
    variance = masked_sq.sum(dim=axis, keepdim=True) / sumofweighting

    if not keep_dims:
        return torch.squeeze(mean, dim=axis), torch.squeeze(variance, dim=axis)
    return mean, variance


class Squeeze(nn.Module):
    def __init__(self, mode='mean'):
        super().__init__()
        self.mode = mode

    def forward(self, input, mask=None):
        if mask is not None:
            mask_weights = mask.type(input.dtype).unsqueeze(dim=-1) + torch.zeros_like(input)
        else:
            mask_weights = torch.ones_like(input)
        m, v = _masked_moments(input, mask_identity=mask_weights, axis=-2, keep_dims=True)
        if self.mode.startswith('mean'):
            return m
        elif self.mode.startswith('var'):
            return v - 1.
        else:
            raise NotImplementedError('Incorrect Squeeze method, permitted values: mean, variance')


class Excitation(nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.
    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """
    def __init__(
            self,
            input_channels: int,
            squeeze_channels: int,
            output_channels: int,
            activation: Callable[..., nn.Module] = None,  # nn.ELU,  # torch.nn.ReLU,
            scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        if activation is None:
            self.fc1 = nn.Linear(input_channels, output_channels)
            self.fc2 = None
            self.activation = None
            # self.fc1.weight.data.copy_(torch.eye(output_channels, input_channels))
            # nn.init.zeros_(self.fc1.bias)
        else:
            self.fc1 = nn.Linear(input_channels, squeeze_channels)
            self.fc2 = nn.Linear(squeeze_channels, output_channels)
            self.activation = activation(inplace=True)
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.fc1(input)
        if self.activation is not None:
            scale = self.activation(scale)
            scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        return self._scale(input)


class CrossAttentionPooling(nn.Module):
    def __init__(self, hid_dim, n_heads, attn_method='dba_l1w', attn0init=False):
        super().__init__()
        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.method = attn_method
        self.scale = np.sqrt(self.head_dim)  # math.sqrt(self.head_dim)

        if self.method.endswith('w'):
            _diag = torch.empty((self.n_heads, 1, self.head_dim))
            if attn0init:
                nn.init.zeros_(_diag)
            else:
                nn.init.ones_(_diag)
            self.register_parameter('attn_diag', torch.nn.Parameter(_diag, requires_grad=True))
        else:
            self.register_buffer('attn_diag', torch.ones((self.n_heads, 1, self.head_dim)))

    def _multiplicative_attention_logits(self, q, k):
        scaled_attention_logit = (self.attn_diag * q) @ k.mT / self.scale  # [batch size, n heads, query len, key len]
        return scaled_attention_logit  # [batch, n_heads, query_len, key_len]

    def _distance_based_attention_logits(self, q, k):
        # memory-efficient dist-based by head
        qdim = q.size(-2)  # (b, h, seq_len_q, ch)
        if self.method.startswith('dba_l1'):
            scaling_factor = np.sqrt(0.72676) * self.scale  # math.sqrt(0.72676) * self.scale
            center_factor = (self.scale ** 2) * 1.12838
        else:
            scaling_factor = np.sqrt(8.0) * self.scale  # math.sqrt(8.) * self.scale
            center_factor = (self.scale ** 2) * 2.0

        qk = []
        for i in range(qdim):
            one_q = q[..., i:(i+1), :]  # (b, h, 1, ch)
            diff = k - one_q  # (b, h, seq_len_k, ch)
            if self.method.startswith('dba_l1'):
                distance = diff.abs_()
            elif self.method.startswith('dba_l2'):
                distance = diff.square_()
            else:
                raise NotImplementedError(f'Unrecognized attention method : {self.method}')
            unscaled_attention_logit = (self.attn_diag * distance).sum(dim=-1, keepdim=False)  # (b, h, k_len)
            scaled_attention_logits = unscaled_attention_logit.sub_(center_factor).div_(-scaling_factor)
            qk.append(scaled_attention_logits)
        attention_logits = torch.stack(qk, dim=-2)  # (b,heads,query_len,key_len)
        return attention_logits  # (batch, n_heads, query_len, key_len)

    def forward(self, attention_q, attention_k, attention_v, mask=None, rel_attn=None, attnlogits_multiplier=1.0):
        expanded_v = (len(attention_v.size()) == 4)
        if mask is not None:
            kdim = mask.size(-1)
            assert kdim == attention_k.size(-2)
            mask = mask.view(-1, 1, 1, kdim)  # [batch, h, qlen, klen]

        split_head = (self.n_heads, self.head_dim)
        Q = attention_q.view(attention_q.size()[:-1] + split_head).transpose(2, 1)
        K = attention_k.view(attention_k.size()[:-1] + split_head).transpose(2, 1)
        V = attention_v.view(attention_v.size()[:-1] + split_head)
        if expanded_v:
            V = V.permute(0, 3, 1, 2, 4)
        else:
            V = V.transpose(2, 1)

        if self.method.startswith('vema') or self.method.startswith('sma'):
            attn_logits = self._multiplicative_attention_logits(Q, K)
        elif self.method.startswith('dba'):
            attn_logits = self._distance_based_attention_logits(Q, K)
        else:
            raise NotImplementedError('Invalid attention method, permitted values: vema, dba_l1/l2, sma')

        attn_logits *= attnlogits_multiplier  # (batch, n_heads, query_len, key_len)
        if mask is not None:
            attn_logits.masked_fill_(mask.logical_not(), float('-inf'))
        if rel_attn is None:
            attention = F.softmax(attn_logits, dim=-1)  # [batch, n_heads, query_len, key_len]
        else:
            attention = rel_normalize(attn_logits, rel_attn)

        if expanded_v:
            x = torch.einsum('...k,...kd->...d', attention, V)  # [batch size, n heads, query len, head dim]
        else:
            x = attention @ V  # [batch size, n heads, query len, head dim]
        x = x.transpose_(2, 1).contiguous()
        output = x.view(x.size()[:-2] + (self.hid_dim,))  # [batch size, query len, hid dim]

        assert output.size() == attention_q.size() or (output.size()[1:] == attention_q.size()[1:] and attention_q.size(0) == 1)
        return output, attention.detach()


def rel_normalize(alogit, adenom):
    aexp = alogit.exp()  # (b,n_heads,q_len,k_len)
    sfmx = aexp / (aexp + adenom)
    return F.normalize(sfmx, p=1, dim=-1)


class CAPBlock(nn.Module):
    def __init__(self, args, input_units, output_units, num_heads, head_init,
                 # attention_method='dba_l1w', skip_conn=False, post_layernorm=True, coexcite=True,
                 coe_intermediate=1.0, squeeze=False, pre_layernorm=False
                 ):
        super().__init__()
        self.channel_size = output_units
        self.skip_conn = args.residual
        self.persist_size = (input_units == output_units)

        if num_heads is None or num_heads <= 0:
            num_heads = 1
            self.vlinear = self.qlinear = self.klinear = self.qslinear = None
        else:
            self.vlinear = self.qslinear = nn.Linear(input_units, output_units)
            if head_init.startswith('id'):
                linid = nn.Linear(input_units, output_units, bias=False)
                linid.weight.data.copy_(torch.eye(output_units, input_units))
                self.qlinear = self.klinear = linid
            else:
                self.qlinear = self.klinear = nn.Linear(input_units, output_units)

        if args.attention_method.startswith('vema'):
            self.var_sqz = Squeeze(mode='variance')
            self.var_exc = Excitation(input_units, input_units, output_units, activation=nn.ELU)
        else:
            self.var_sqz = self.var_exc = None

        self.coe_sqz = self.coe_excite = None
        if args.excite_param:
            self.coe_excite = Excitation(
                input_channels=input_units,
                squeeze_channels=int(output_units * coe_intermediate),
                output_channels=output_units,
                activation=None  # nn.ReLU  # nn.ELU
            )
            if squeeze:
                self.coe_sqz = Squeeze(mode='mean')

        self.cap = CrossAttentionPooling(
            output_units,
            n_heads=num_heads,
            attn_method=args.attention_method,
        )

        self.pre_layernorm = nn.LayerNorm(input_units,
                                          # elementwise_affine=False,
                                          ) if pre_layernorm else None
        self.post_layernorm = None

        self.lg = None  # nn.Linear(input_units, output_units)
        self.glu = None  # nn.GLU()

    def forward(self, query_input, bag_input, bag_mask=None, attnlogits_multiplier=1.0,
                excite_sqzed=None, relative_attn=None):
        if self.pre_layernorm:
            query = self.pre_layernorm(query_input)
            bag = self.pre_layernorm(bag_input)
        else:
            query = query_input
            bag = bag_input

        if self.qlinear:
            attn_q = self.qlinear(query)  # (b,q,ch)
        else:
            attn_q = query
        if self.klinear:
            k = self.klinear(bag)  # (b,k,ch)
        else:
            k = bag
        if self.vlinear:
            v = self.vlinear(bag)  # (b,k,ch)
        else:
            v = bag
        if self.qslinear:
            query_branch = self.qslinear(query)  # (b,q,ch)
        else:
            query_branch = query

        # variance excitation
        if self.var_sqz is not None and self.var_exc is not None:
            diag = self.var_exc(self.var_sqz(bag))
        else:
            diag = 1.0
        attn_k = k * diag

        # co-excitation
        if self.coe_excite is not None:
            if excite_sqzed is not None:
                sqz = excite_sqzed
            elif self.coe_sqz is not None:
                sqz = self.coe_sqz(query)  # (b,1,ch)
            else:
                sqz = query  # (b,q,ch)
                # print('query excited, no squeeze', flush=True)

            co_excite = self.coe_excite(sqz)  # (b, 1|q, ch)
            if self.skip_conn:
                excited_query = query_branch * co_excite + attn_q
            else:
                excited_query = query_branch * co_excite  # (b,q,ch)
            if co_excite.size(-2) > 1:  # (b,q,ch)
                co_excite = co_excite.unsqueeze(-2)
                v = v.unsqueeze(-3)
                addk = k.unsqueeze(-3)
            else:  # broadcastable
                addk = k
            # (b,q,k,ch)
            if self.skip_conn:
                excited_v = v * co_excite + addk
            else:
                excited_v = v * co_excite
        else:
            excited_v = v
            excited_query = query_branch

        # CA Pooling
        ca_pool, detached_as = self.cap(attn_q, attn_k, excited_v, mask=bag_mask,
                                        attnlogits_multiplier=attnlogits_multiplier)  # (b,q,ch)

        # post layernorm
        if self.post_layernorm:
            output_qry = self.post_layernorm(excited_query)
            output_bag = self.post_layernorm(ca_pool)  # (b,q,ch)
        else:
            output_qry = excited_query
            output_bag = ca_pool

        return output_qry, output_bag, detached_as


class PlainTaskDataset(data.Dataset):
    def __init__(
            self,
            tasks,
            transform,
            aux_transform=None
    ):
        self.tasks = tasks
        self.loader = default_loader
        self.transform = transform
        self.aux_transform = aux_transform

    def list2tensor(self, instlist, use_auxt=False):
        output = []
        for il in instlist:
            sample = self.loader(il)
            if use_auxt and self.aux_transform is not None:
                output.append(self.aux_transform(sample))
            else:
                output.append(self.transform(sample))
        return torch.stack(output, dim=0)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Task Index

        Returns:
            tuple: (query_set, support_set, query_class)
        """
        task = self.tasks[index]
        query_set = self.list2tensor(task['query'])
        supportset = []
        for s in task['support']:
            stensor = self.list2tensor(s)
            supportset.append(stensor)
        target = torch.as_tensor(task['qcidx'])
        return query_set, supportset, target

    def __len__(self) -> int:
        return len(self.tasks)


class TaskDataset(data.Dataset):
    def __init__(
            self,
            args,
            tasks,
            transform,
            aux_transform=None,
            distort_transform=None
    ):
        self.tasks = tasks
        self.loader = default_loader
        self.transform = transform
        self.aux_transform = aux_transform
        self.distort_transform = distort_transform
        # self.resolution = args.resolution
        self.num_sda = args.num_da

    def list2tensor(self, instlist, use_auxt=False, distort=0):
        output = []
        add_transform = self.aux_transform if use_auxt else self.distort_transform
        if distort > 0:
            assert add_transform is not None

        for il in instlist:
            sample = self.loader(il)
            if distort > 0:
                for j in range(distort):
                    output.append(add_transform(sample))
            else:
                output.append(self.transform(sample))
        return torch.stack(output, dim=0)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        task = self.tasks[index]
        query_set = self.list2tensor(task['query'])

        # nways = len(task['support'])
        support_set = []
        test_supportset = []
        for s in task['support']:
            nshots = len(s)
            stensor = self.list2tensor(s)
            test_supportset.append(stensor)

            if self.aux_transform is not None:
                s1c = self.list2tensor(s, use_auxt=True)
            else:
                s1c = stensor
            if nshots <= self.num_sda:
                distorted = self.list2tensor(s, distort=int(np.ceil(self.num_sda / nshots)))
                s1c = torch.cat([s1c, distorted], dim=0)
            support_set.append(s1c)

        target = torch.as_tensor(task['qcidx'])
        return query_set, (test_supportset, support_set), target

    def __len__(self) -> int:
        return len(self.tasks)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=None):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if topk is None:
        topk = (1,)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

