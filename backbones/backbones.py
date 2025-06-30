# Copyright (c) 2025 X. Xu; All Rights Reserved.

import argparse
import numpy as np
import os
from functools import partial

from torch import nn, optim
import torch
import torch.nn.functional as F
import torchvision
torchvision.disable_beta_transforms_warning()

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='timm')
import timm
# from timm.models.vision_transformer import VisionTransformer, _cfg
from . import DINO_ViT as dino, resnet18sdl as rnet18

# Additional backbones: SimMIM(SwinT-B); CLIP(ViT-l/14); DINO(ViT-b/8); DeiT-b/16; RegNetY-1.6GF; DenseNet-161
from .swin_transformer import get_config, build_swin
# pip install safetensors==0.4.3
from transformers import CLIPVisionModel
# timm, huggingface


class DINOBackbone(nn.Module):
    def __init__(self, args):  # vit_arch, patch_size, pretrained, concat_gap=True, last_n=1
        super().__init__()
        if args.vit_arch.startswith('deit'):
            self.nnmodel = dino.vit_small(args.patch_size)
        else:
            self.nnmodel = dino.vit_base(args.patch_size)
        dino.load_pretrained_weights(
            model=self.nnmodel,
            pretrained_weights=args.pretrained,
            patch_size=args.patch_size,
            model_name=None,
            checkpoint_key=None
        )
        self.last_nblocks = args.n_outputs
        self.patch_size = args.patch_size
        self.maxpool_sizes = args.maxpool_sizes
        self.nnmodel.eval()

    def forward(self, input, mode=None):  # (b,3,H,W)
        intermediate_output = self.nnmodel.get_intermediate_layers(input, n=self.last_nblocks)
        _sizes = self.maxpool_sizes[(-self.last_nblocks):]
        output_list = []
        for i, x in enumerate(intermediate_output):
            raw_output = x[:, 1:]  # (B,p,C)
            cls = x[:, 0:1]  # (B,1,C)
            pooled_output = [cls]

            batch, sq, channel = raw_output.size()
            _hw = int(np.sqrt(sq))
            reshaped_output = raw_output.mT.view(batch, channel, _hw, _hw)
            for _s in _sizes[i]:
                _spp = F.adaptive_max_pool2d(reshaped_output, _s).view(batch, channel, -1).mT
                pooled_output.append(_spp)  # [(B,P,C)]
            output_list.append(pooled_output)
        return output_list

    def pooled_output(self, input):
        return self.nnmodel(input)


# supervised backbone: DeiT
class DeiT16Backbone(nn.Module):
    def __init__(self, args):  # vit_arch, patch_size, pretrained, concat_gap=True, last_n=1
        super().__init__()
        assert args.patch_size == 16
        nheads = args.channel_size // 64
        self.nnmodel = timm.models.vision_transformer.VisionTransformer(
            patch_size=args.patch_size, embed_dim=args.channel_size, depth=12, num_heads=nheads, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0
        )
        self.nnmodel.default_cfg = timm.models.vision_transformer._cfg()
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        msg = self.nnmodel.load_state_dict(checkpoint['model'], strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained, msg))

        self.last_nblocks = args.n_outputs
        self.patch_size = args.patch_size
        self.maxpool_sizes = args.maxpool_sizes
        self.nnmodel.eval()

    def forward(self, input, mode=None):  # (b,3,H,W)
        intermediate_output = self.nnmodel.get_intermediate_layers(
            input,
            n=self.last_nblocks,
            return_prefix_tokens=True,
            norm=True
        )

        _sizes = self.maxpool_sizes[(-self.last_nblocks):]
        output_list = []
        for i, (raw_output, cls) in enumerate(intermediate_output):  # (B,p,C)(B,1,C)
            pooled_output = [cls]
            batch, sq, channel = raw_output.size()
            _hw = int(np.sqrt(sq))
            reshaped_output = raw_output.mT.view(batch, channel, _hw, _hw)
            for _s in _sizes[i]:
                _spp = F.adaptive_max_pool2d(reshaped_output, _s).view(batch, channel, -1).mT
                pooled_output.append(_spp)  # [(B,P,C)]
            output_list.append(pooled_output)
        return output_list

    def pooled_output(self, input):
        return self.nnmodel(input)


def pooling(_x, _size):
    if _size is not None:
        _y = F.adaptive_max_pool2d(_x, _size)
    else:
        _y = _x
    return torch.flatten(F.adaptive_avg_pool2d(_y, (1, 1)), 1)


# supervised backbone: ResNets
class RNetBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.rnet_modelname.startswith('dino'):
            self.nnmodel = torchvision.models.resnet50(weights=None)
            state_dict = torch.load(args.rnet_pretrained, map_location='cpu')
            msg = self.nnmodel.load_state_dict(state_dict, strict=False)
            print('DINO-pretrained weights loaded with msg: {}'.format(msg), flush=True)
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
        self.model_name = args.rnet_modelname
        # 'IMAGENET1K_V1', None
        print(self.model_name, 'pretrained_weights =', args.rnet_pretrained, flush=True)

        total_blocks = [self.nnmodel.layer4, self.nnmodel.layer3, self.nnmodel.layer2, self.nnmodel.layer1]
        assert args.rnet_lastn_blocks <= len(total_blocks)
        self.lastn_blocks = total_blocks[:args.rnet_lastn_blocks]
        self.maxpool_sizes = args.maxpool_sizes
        self.nnmodel.eval()

    def forward(self, input, mode=None):
        intermediate_output = {}
        hook_handlers = []
        total_outputs = len(self.lastn_blocks)
        def get_intermediate(layer_name):
            def hook(module, input, output):
                intermediate_output[layer_name] = output
            return hook

        for i, b in enumerate(self.lastn_blocks):
            handler = b.register_forward_hook(get_intermediate(f'block{total_outputs - i}'))
            hook_handlers.append(handler)
        with torch.no_grad():
            self.nnmodel(input)
        for h in hook_handlers:
            h.remove()

        _sizes = self.maxpool_sizes[(-total_outputs):]
        output_list = []
        for i, key in enumerate(intermediate_output):  # ascending
            y = intermediate_output[key]
            batch, channel, _h, _w = y.size()

            mps = [y.view(batch, channel, -1)]
            for s in _sizes[i]:
                _input = F.adaptive_max_pool2d(y, s).view(batch, channel, -1)
                mps.append(_input)
            output_list.append(mps)
        return output_list

    def pooled_output(self, input):
        return self.nnmodel(input)


# special backbone: SDL-ResNet18
class SDLRNet18Backbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.nnmodel = rnet18.resnet18(pretrained_model_path=args.rnet_pretrained, num_classes=512, classifier='linear')
        total_blocks = [self.nnmodel.layer4, self.nnmodel.layer3, self.nnmodel.layer2, self.nnmodel.layer1]
        assert args.rnet_lastn_blocks <= len(total_blocks)
        self.lastn_blocks = total_blocks[:args.rnet_lastn_blocks]
        self.maxpool_sizes = args.maxpool_sizes
        self.nnmodel.eval()

    def forward(self, input, mode=None):
        intermediate_output = {}
        hook_handlers = []
        total_outputs = len(self.lastn_blocks)
        def get_intermediate(layer_name):
            # print('get intermediate')
            def hook(module, input, output):
                intermediate_output[layer_name] = output
            return hook

        for i, b in enumerate(self.lastn_blocks):
            handler = b.register_forward_hook(get_intermediate(f'block{total_outputs - i}'))
            hook_handlers.append(handler)
        with torch.no_grad():
            self.nnmodel(input)
        for h in hook_handlers:
            h.remove()

        _sizes = self.maxpool_sizes[(-total_outputs):]
        output_list = []
        for i, key in enumerate(intermediate_output):  # ascending
            y = intermediate_output[key]
            batch, channel, _h, _w = y.size()

            mps = [y.view(batch, channel, -1)]
            for s in _sizes[i]:
                _input = F.adaptive_max_pool2d(y, s).view(batch, channel, -1)
                mps.append(_input)
            output_list.append(mps)
        return output_list


def compat_resize(input_tensor, size):  # (C,H,W)
    output = F.interpolate(
        input_tensor.unsqueeze(0),
        size=size,
        mode='bilinear',
        align_corners=True
    )
    return output.squeeze_(0).to(dtype=torch.get_default_dtype()).div_(255.0)


class CNNBackbone(nn.Module):
    def __init__(self, args, intermediate=False):
        super().__init__()
        self.intermediate = intermediate
        if self.intermediate:
            self.nnmodel = timm.create_model(args.rnet_pretrained, pretrained=True, features_only=True)
        else:
            self.nnmodel = timm.create_model(args.rnet_pretrained, pretrained=True, num_classes=0)
        print(args.rnet_modelname, ': pretrained_weights =', args.rnet_pretrained, flush=True)
        self.lastn_blocks = args.rnet_lastn_blocks
        self.maxpool_sizes = args.maxpool_sizes
        self.nnmodel.eval()

    def forward(self, input):
        assert self.intermediate
        intermediates = self.nnmodel(input)
        intermediate_output = intermediates[(-self.lastn_blocks):]
        _sizes = self.maxpool_sizes[(-self.lastn_blocks):]
        output_list = []
        for i, y in enumerate(intermediate_output):
            batch, channel, _h, _w = y.size()
            mps = [y.view(batch, channel, -1)]
            for s in _sizes[i]:
                _input = F.adaptive_max_pool2d(y, s).view(batch, channel, -1)
                mps.append(_input)
            output_list.append(mps)
        return output_list

    def pooled_output(self, input):
        assert (not self.intermediate)
        return self.nnmodel(input)


class ClipBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 'openai/clip-vit-large-patch14'
        self.nnmodel = CLIPVisionModel.from_pretrained(args.pretrained)
        msg = self.nnmodel.vision_model.config
        print('Pretrained weights found at {} and loaded with config: {}'.format(args.pretrained, msg))
        self.last_nblocks = args.n_outputs
        self.maxpool_sizes = args.maxpool_sizes

    def forward(self, input):  # (b,3,H,W)
        hidden_states = self.nnmodel(input, output_hidden_states=True).hidden_states
        intermediate_output = hidden_states[(-self.last_nblocks):]
        _sizes = self.maxpool_sizes[(-self.last_nblocks):]

        output_list = []
        for i, raw_output in enumerate(intermediate_output):  # (B,1+p,C)
            # (B,1,C)(B,p,C)
            cls = raw_output[:, 0:1]
            tokens = raw_output[:, 1:]
            pooled_output = [cls]  # []  #
            batch, sq, channel = tokens.size()
            _hw = int(np.sqrt(sq))
            reshaped_output = tokens.mT.view(batch, channel, _hw, _hw)
            for _s in _sizes[i]:
                _spp = F.adaptive_max_pool2d(reshaped_output, _s).view(batch, channel, -1).mT
                pooled_output.append(_spp)  # [(B,P,C)]
            output_list.append(pooled_output)
        return output_list

    def pooled_output(self, input):
        return self.nnmodel(input).pooler_output


class SWTBackbone(nn.Module):
    def __init__(self, args, post_norm=False):
        super().__init__()
        # 'simmim_finetune__swin_base__img224_window7__800ep'
        swt_args = argparse.ArgumentParser('Swin Transformer', add_help=False).parse_args([])
        swt_args.cfg = args.pretrained + '.yaml'
        swt_args.opts = None
        swt_args.local_rank = False
        config = get_config(swt_args)
        self.nnmodel = build_swin(config)
        self.post_norm = post_norm

        state_dict = torch.load(args.pretrained + '.pth', map_location="cpu")
        msg = self.nnmodel.load_state_dict(state_dict['model'], strict=False)
        print('Pretrained weights (post-norm={}) found at {} and loaded with msg: {}'.format(self.post_norm, args.pretrained, msg))
        self.last_nblocks = args.n_outputs
        self.maxpool_sizes = args.maxpool_sizes

    def forward(self, input):  # (b,3,H,W)
        _, hidden_layers = self.nnmodel.forward_features(input)
        intermediate_output = hidden_layers[(-self.last_nblocks):]
        _sizes = self.maxpool_sizes[(-self.last_nblocks):]

        output_list = []
        for i, tokens in enumerate(intermediate_output):
            pooled_output = []
            batch, sq, channel = tokens.size()
            raw_output = tokens
            # swin-transformer lacks output norm in intermediate layers
            if self.post_norm:
                raw_output = F.layer_norm(raw_output, [channel])
            _hw = int(np.sqrt(sq))
            reshaped_output = raw_output.mT.view(batch, channel, _hw, _hw)
            for _s in _sizes[i]:
                if _s == _hw:
                    pooled_output.append(raw_output)
                else:
                    _spp = F.adaptive_max_pool2d(reshaped_output, _s).view(batch, channel, -1).mT
                    pooled_output.append(_spp)  # [(B,P,C)]
            output_list.append(pooled_output)
        return output_list

    def pooled_output(self, input):
        pooled, _ = self.nnmodel.forward_features(input)
        return pooled


# A helper class for standard classifiers
class Backbone(nn.Module):
    def __init__(self, args):
        super(Backbone, self).__init__()
        self.special_embed = False
        if args.backbone.startswith('swint'):
            self.backbone = SWTBackbone(args, post_norm=True)
        elif 'clip' in args.backbone:
            self.backbone = ClipBackbone(args)
        elif args.backbone.startswith(('vit', 'deit')):
            if 'dino' in args.backbone:
                self.backbone = DINOBackbone(args)
            else:
                self.backbone = DeiT16Backbone(args)
        elif args.backbone.startswith('cnn'):
            self.backbone = CNNBackbone(args)
        elif args.backbone.startswith('sdl'):
            self.backbone = SDLRNet18Backbone(args)
            self.special_embed = True
        else:
            self.backbone = RNetBackbone(args)

    def forward(self, input):
        self.backbone.eval()
        if self.special_embed:
            output = self.backbone.embed(input)  # (Q,C)
        else:
            output = self.backbone.pooled_output(input)
        return output
