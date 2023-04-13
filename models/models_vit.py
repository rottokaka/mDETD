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
import copy

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch import Tensor

import timm.models.vision_transformer

from util.misc import NestedTensor, is_main_process

from typing import Tuple, List, Dict, Optional


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, use_simple_fpn=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.in_chans = kwargs['in_chans']
        del self.norm  # remove the original norm
        self.use_simple_fpn = use_simple_fpn

        if not self.use_simple_fpn:
            self.simple_fpn = torchvision.ops.FeaturePyramidNetwork([768,768,768,768], 256)
        else:
            self.simple_fpn = SimpleFPN([768], 256)

        self.norm_ = nn.LayerNorm(256)
        self.linear_ = nn.Linear(768, 256)
        self.origin_size = [kwargs['img_size'], kwargs['img_size']]
        self._init_patch_embed()

    def _init_patch_embed(self):
        self.patch_embed = PatchEmbed(
            img_size=224, patch_size=16, in_chans=3, embed_dim=768)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

    # def _interpolate_pos_embed(self, img_size):
    #     if self.origin_size != img_size

    def forward_features(self, tensor_list: NestedTensor, position_embedding, position_embedding_):
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
        
        if not self.use_simple_fpn:
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
        else:
            for _, blk in enumerate(self.blocks):
                x = blk(x)
            xs['block'+str(_)] = x[:,1:,:].permute(0,2,1).view(B, -1, H//16, W//16)
            xs = self.simple_fpn(xs)
            xs = list(xs.values())
            for x in xs:
                H_, W_ = x.shape[2:]
                masks.append(F.interpolate(mask[None].float(), size=(H_,W_)).to(torch.bool)[0])

        pos_embeds = [position_embedding(NestedTensor(xs[i], masks[i])) for i in range(len(xs))]

        return xs, masks, pos_embeds
    
    def forward(self, tensor_list: NestedTensor, position_embedding, position_embedding_):
        self.to('cuda:0')
        xs, masks, pos_embeds = self.forward_features(tensor_list, position_embedding, position_embedding_)
        return xs, masks, pos_embeds
    
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class ExtraFPNBlock(nn.Module):
    """
    Base class for the extra block in the FPN.

    Args:
        results (List[Tensor]): the result of the FPN
        x (List[Tensor]): the original feature maps
        names (List[str]): the names for each one of the
            original feature maps

    Returns:
        results (List[Tensor]): the extended set of results
            of the FPN
        names (List[str]): the extended set of names for the results
    """
    def forward(
        self,
        results: List[Tensor],
        x: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        pass

class SimpleFPN(torchvision.ops.FeaturePyramidNetwork):
    def __init__(self, in_channels_list: List[int], out_channels: int, extra_blocks: Optional[ExtraFPNBlock] = None):
        super().__init__(in_channels_list, out_channels, extra_blocks)
        self.simple_blocks_0 = nn.ModuleList()
        self.simple_blocks_1 = nn.ModuleList()
        self.simple_blocks_2 = nn.ModuleList()
        self.simple_blocks_3 = nn.ModuleList()
        self.layer_norm = nn.LayerNorm(out_channels)
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            simple_block_module_0 = nn.Conv2d(in_channels, out_channels, 1, 2)
            simple_block_module_1 = nn.Conv2d(in_channels, out_channels, 1, 1)
            simple_block_module_2 = nn.ConvTranspose2d(in_channels, out_channels, 1, 2)
            simple_block_module_3 = nn.ConvTranspose2d(in_channels, out_channels, 1, 4)
            self.simple_blocks_0.append(simple_block_module_0)
            self.simple_blocks_1.append(simple_block_module_1)
            self.simple_blocks_2.append(simple_block_module_2)
            self.simple_blocks_3.append(simple_block_module_3)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        if extra_blocks is not None:
            assert isinstance(extra_blocks, ExtraFPNBlock)
        self.extra_blocks = extra_blocks

    def get_simple_blocks(self, block_idx):
        if block_idx == 1:
            return self.simple_blocks_1
        elif block_idx == 2:
            return self.simple_blocks_2
        elif block_idx == 3:
            return self.simple_blocks_3
        return self.simple_blocks_0

    def get_result_from_simple_blocks(self, x: Tensor, idx: int, block_idx: int, output_size=None) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.simple_blocks_0)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.get_simple_blocks(block_idx):
            if i == idx and output_size is None:
                out = module(x)
            elif i == idx and output_size is not None:
                out = module(x, output_size)
            i += 1
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.
        Extra block is not supported yet

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        results = []
        for idx in range(len(x)):
            H, W = x[idx].shape[2:]
            out_0 = self.layer_norm(self.get_result_from_simple_blocks(x[idx], idx, 0).permute(0,2,3,1)).permute(0,3,1,2)
            out_1 = self.layer_norm(self.get_result_from_simple_blocks(x[idx], idx, 1).permute(0,2,3,1)).permute(0,3,1,2)
            out_2 = self.layer_norm(self.get_result_from_simple_blocks(x[idx], idx, 2, [H*2, W*2]).permute(0,2,3,1)).permute(0,3,1,2)
            out_3 = self.layer_norm(self.get_result_from_simple_blocks(x[idx], idx, 3, [H*4, W*4]).permute(0,2,3,1)).permute(0,3,1,2)
            results.append([out_0, out_1, out_2, out_3])

        out = OrderedDict()
        # make it back an OrderedDict
        for idx in range(len(x)):
            for block_idx in range(4):
                out[names[idx]+'_'+str(block_idx)] = results[idx][block_idx]

        return out

    

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
