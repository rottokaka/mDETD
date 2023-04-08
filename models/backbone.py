# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from timm.models.vision_transformer import HybridEmbed, PatchEmbed
from util.pos_embed import interpolate_pos_embed
from models import models_vit


from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding, build_position_encoding_


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class ViTBackbone(nn.Module):
    def __init__(self, args, position_embedding, position_embedding_):
        super().__init__()
        if args.backbone == "vit_base_patch16":
            backbone = models_vit.__dict__['vit_base_patch16'](
                drop_path_rate=0.1,
                in_chans=3
            )
            if args.pretrained_backbone_path:
                checkpoint = torch.load(args.pretrained_backbone_path, map_location='cpu')
                checkpoint_backbone = checkpoint['model']
                state_dict = backbone.state_dict()
                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint_backbone and checkpoint_backbone[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_backbone[k]
                interpolate_pos_embed(backbone, checkpoint_backbone)
                backbone.load_state_dict(checkpoint_backbone, strict=False)
            if args.poor:
                for _, block in enumerate(backbone.blocks):
                    if _ < 8:
                        for param in block.parameters():
                            param.requires_grad = False
            self.backbone = backbone
            self.position_embedding = position_embedding
            self.position_embedding_ = position_embedding_
        else:
            raise NotImplementedError
        
    def forward(self, tensor_list: NestedTensor):
        srcs, masks, pos_embeds = self.backbone(tensor_list, self.position_embedding, self.position_embedding_)
        return srcs, masks, pos_embeds

# class TransformerBackbone(nn.Module):
#     def __init__(
#         self, backbone: str, train_backbone: bool, return_interm_layers: bool, args
#     ):
#         super().__init__()
#         out_indices = (1, 2, 3)
#         if backbone == "swin_tiny":
#             backbone = SwinTransformer(
#                 embed_dim=96,
#                 depths=[2, 2, 6, 2],
#                 num_heads=[3, 6, 12, 24],
#                 window_size=7,
#                 ape=False,
#                 drop_path_rate=args.drop_path_rate,
#                 patch_norm=True,
#                 use_checkpoint=True,
#                 out_indices=out_indices,
#             )
#             embed_dim = 96
#             backbone.init_weights(args.pretrained_backbone_path)
#         elif backbone == "swin_small":
#             backbone = SwinTransformer(
#                 embed_dim=96,
#                 depths=[2, 2, 18, 2],
#                 num_heads=[3, 6, 12, 24],
#                 window_size=7,
#                 ape=False,
#                 drop_path_rate=args.drop_path_rate,
#                 patch_norm=True,
#                 use_checkpoint=True,
#                 out_indices=out_indices,
#             )
#             embed_dim = 96
#             backbone.init_weights(args.pretrained_backbone_path)
#         elif backbone == "swin_large":
#             backbone = SwinTransformer(
#                 embed_dim=192,
#                 depths=[2, 2, 18, 2],
#                 num_heads=[6, 12, 24, 48],
#                 window_size=7,
#                 ape=False,
#                 drop_path_rate=args.drop_path_rate,
#                 patch_norm=True,
#                 use_checkpoint=True,
#                 out_indices=out_indices,
#             )
#             embed_dim = 192
#             backbone.init_weights(args.pretrained_backbone_path)
#         elif backbone == "swin_large_window12":
#             backbone = SwinTransformer(
#                 pretrain_img_size=384,
#                 embed_dim=192,
#                 depths=[2, 2, 18, 2],
#                 num_heads=[6, 12, 24, 48],
#                 window_size=12,
#                 ape=False,
#                 drop_path_rate=args.drop_path_rate,
#                 patch_norm=True,
#                 use_checkpoint=True,
#                 out_indices=out_indices,
#             )
#             embed_dim = 192
#             backbone.init_weights(args.pretrained_backbone_path)
#         else:
#             raise NotImplementedError

#         for name, parameter in backbone.named_parameters():
#             # TODO: freeze some layers?
#             if not train_backbone:
#                 parameter.requires_grad_(False)

#         if return_interm_layers:

#             self.strides = [8, 16, 32]
#             self.num_channels = [
#                 embed_dim * 2,
#                 embed_dim * 4,
#                 embed_dim * 8,
#             ]
#         else:
#             self.strides = [32]
#             self.num_channels = [embed_dim * 8]

#         self.body = backbone

#     def forward(self, tensor_list: NestedTensor):
#         xs = self.body(tensor_list.tensors)

#         out: Dict[str, NestedTensor] = {}
#         for name, x in xs.items():
#             m = tensor_list.mask
#             assert m is not None
#             mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
#             out[name] = NestedTensor(x, mask)
#         return out

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    position_embedding_ = build_position_encoding_(args)
    backbone = ViTBackbone(args, position_embedding, position_embedding_)
    return backbone
