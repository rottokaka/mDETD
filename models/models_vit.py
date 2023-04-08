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

import torch
import torch.nn as nn

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

        self.return_layers = {"block8": "0", "block9": "1", "block10": "2", "block11": "3"}
        self.linear = nn.Linear(728,256)

    def _init_patch_embed(self, img_size):
        if self.hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                self.hybrid_backbone, img_size=img_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

    def forward_features(self, x):#: NestedTensor):
        # x = tensor_list.tensors
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x) 
        xs = []

        for _, blk in enumerate(self.blocks):
            x = blk(x)
            if _ in [8, 9,10,11]:
                xs.append(x)

        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out

        return xs
    
    def forward(self, tensor_list):
        # self._init_patch_embed(x.shape[2:])
        self._init_patch_embed(224)
        self.to('cuda:0')
        outs = self.forward_features(tensor_list)
        return outs


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
