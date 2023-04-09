# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import timm.models.vision_transformer
from timm.models.vision_transformer import HybridEmbed, PatchEmbed

from util.misc import NestedTensor, is_main_process


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.in_chans = kwargs['in_chans']
        del self.norm  # remove the original norm

        self.simple_fpn = torchvision.ops.FeaturePyramidNetwork([768,768,768,768], 256)
        self.norm_ = nn.LayerNorm(256)
        self.linear_ = nn.Linear(768, 256)

    def _init_patch_embed(self, img_size):
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

    def forward_features(self, tensor_list: NestedTensor, position_embedding, position_embedding_):#: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        B = x.shape[0]
        H, W = x.shape[2:]
        H_, W_ = H//16, W//16
        x = self.patch_embed(x)
        mask = F.interpolate(mask[None].float(), size=(H_,W_)).to(torch.bool)[0]
        mask_ = F.interpolate(mask[None].float(), size=(H_,W_)).to(torch.bool)[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        mask_ = F.interpolate(mask[None].float(), size=(x.shape[1],1)).to(torch.bool)[0]
        pos = position_embedding_(NestedTensor(x, mask_))
        if pos is not None:
            x = x + pos.squeeze(3).permute(0,2,1)
        x = self.pos_drop(x)
        xs = OrderedDict()
        masks = []
        

        for _, blk in enumerate(self.blocks):
            x = blk(x)
            if _ in [10,11]:
                xs['block'+str(_)] = x[:,1:,:].permute(0,2,1).view(B, -1, H//16, W//16)
                masks.append(mask)

        xs_ = self.simple_fpn(xs)
        xs['block10_'] = xs_['block10']
        xs['block11_'] = xs_['block11']
        xs['block10'] = self.linear_(xs['block10'].permute(0,2,3,1)).permute(0,3,1,2)
        xs['block11'] = self.linear_(xs['block11'].permute(0,2,3,1)).permute(0,3,1,2)
        xs = list(xs.values())
        xs = [self.norm_(x.permute(0,2,3,1)).permute(0,3,1,2) for x in xs]
        masks.extend(masks)
        pos_embeds = [position_embedding(NestedTensor(xs[i], masks[i])) for i in range(len(xs))]

        return xs, masks, pos_embeds
    
    def forward(self, tensor_list: NestedTensor, position_embedding, position_embedding_):
        self._init_patch_embed(tensor_list.tensors.shape[2:])
        # self._init_patch_embed(224)
        self.to('cuda:0')
        xs, masks, pos_embeds = self.forward_features(tensor_list, position_embedding, position_embedding_)
        return xs, masks, pos_embeds


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
