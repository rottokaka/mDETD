# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
from models.position_encoding import build_position_encoding_, build_position_encoding
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model, models_vit
from util.pos_embed import interpolate_pos_embed
from datasets.data_prefetcher import data_prefetcher
import torchvision


def get_args_parser(mode="debug", version="0"):
    parser = argparse.ArgumentParser('ViT Deformable DETR Detector', add_help=False)
    parser.add_argument('--mode', default=mode, type=str)
    parser.add_argument('--version', default=version, type=str)
    parser.add_argument('--lr', default=1e-6, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.backbone"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-6, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--poor', default=True, action='store_true',
                        help="This will freeze parameters of first 8 block of vit")
    if version == "2":
        parser.add_argument('--use_simple_fpn', default=True, action='store_true',
                            help="Determine if simple fpn is used") 
    else:
        parser.add_argument('--use_simple_fpn', default=False, action='store_true',
                            help="Determine if simple fpn is used") 
    if version == "3" or version == "4":
        parser.add_argument('--backbone', default='resnet50', type=str,
                            help="Name of the backbone to use")
    else:
        parser.add_argument('--backbone', default='vit_base_patch16', type=str,
                            help="Name of the backbone to use")
    parser.add_argument('--pretrained_backbone_path', default='', type=str,
                        help="Path to the pretrained backbone")
    # parser.add_argument('--pretrained_backbone_path', default='./pre-trained checkpoints/mae_pretrain_vit_base.pth', type=str,
    #                     help="Path to the pretrained backbone")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    if mode == "debug":
        parser.add_argument('--dataset_file', default='coco')
    elif mode == "train" or mode == "val":
        parser.add_argument('--dataset_file', default='minicoco')
    else:
        parser.add_argument('--dataset_file', default='chvg')
    if mode == "demo":
        parser.add_argument('--num_classes', default=9, type=int)
    else:
        parser.add_argument('--num_classes', default=91, type=int)
    parser.add_argument('--coco_path', default='detr//datasets//', type=str)
    parser.add_argument('--mini_coco_path_train_img', default='/kaggle/input/coco25k/images', type=str)
    parser.add_argument('--mini_coco_path_train_ann', default='/kaggle/input/minicoco-annotations/instances_minitrain2017.json', type=str)
    parser.add_argument('--mini_coco_path_val_img', default='/kaggle/input/coco-2017-dataset/coco2017/val2017', type=str)
    parser.add_argument('--mini_coco_path_val_ann', default='/kaggle/input/coco-2017-dataset/coco2017/annotations/instances_val2017.json', type=str)
    parser.add_argument('--chvg_path_train_img', default='/kaggle/input/chvg-coco-format/CHVG_Coco_arg1/train2017', type=str)
    parser.add_argument('--chvg_path_train_ann', default='/kaggle/input/chvg-coco-format/CHVG_Coco_arg1/annotations/instances_train2017.json', type=str)
    if mode == "finetune":
        parser.add_argument('--chvg_path_val_img', default='/kaggle/input/chvg-coco-format/CHVG_Coco_arg1/test2017', type=str)
    elif mode == "finetune0":
        parser.add_argument('--chvg_path_val_img', default='/kaggle/input/chvg-coco-format/CHVG_Coco_arg1/test2017_low_light', type=str)
    elif mode == "finetune1":
        parser.add_argument('--chvg_path_val_img', default='/kaggle/input/chvg-coco-format/CHVG_Coco_arg1/test2017_haze', type=str)
    elif mode == "finetune2":
        parser.add_argument('--chvg_path_val_img', default='/kaggle/input/chvg-coco-format/CHVG_Coco_arg1/test2017_rain', type=str)
    parser.add_argument('--chvg_path_val_ann', default='/kaggle/input/chvg-coco-format/CHVG_Coco_arg1/annotations/instances_test2017.json', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='../',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    if mode == "train" or mode == "val" or mode == "finetune" or mode == "finetune0" or mode == "finetune1" or mode == "finetune2":
        parser.add_argument('--resume', default='/kaggle/working/mDETD_'+version+'.pth', help='resume from checkpoint')
    elif mode == "debug":
        if version == "3":
            parser.add_argument('--resume', default='pre-trained checkpoints/r50_deformable_detr-checkpoint.pth', help='resume from checkpoint')
        elif version == "4":
            parser.add_argument('--resume', default='pre-trained checkpoints/full_coco_finetune.pth', help='resume from checkpoint')
        else:
            parser.add_argument('--resume', default='pre-trained checkpoints/mDETD_'+version+'.pth', help='resume from checkpoint')
        # parser.add_argument('--resume', default='', help='resume from checkpoint')
    elif mode == "demo":
        # model_0
        if version == "0":
            parser.add_argument('--resume', default='pre-trained checkpoints/mDETD_0_.pth', help='resume from checkpoint')
        # model_1
        elif version == "1":
            parser.add_argument('--resume', default='pre-trained checkpoints/mDETD_1_.pth', help='resume from checkpoint')
        # model_2
        elif version == "2":
            parser.add_argument('--resume', default='pre-trained checkpoints/mDETD_2_.pth', help='resume from checkpoint')
        # DDETR
        elif version == "3":
            parser.add_argument('--resume', default='pre-trained checkpoints/DDETR_.pth', help='resume from checkpoint')
        # DETReg
        elif version == "4":
            parser.add_argument('--resume', default='pre-trained checkpoints/DETReg_.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    mode = args.mode
    version = args.version

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters)
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    # for n, p in model_without_ddp.named_parameters():
    #     print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and not match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]

    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['backbone.backbone.pos_embed']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                del checkpoint_model[k]
        interpolate_pos_embed(model_without_ddp, checkpoint_model)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        # print(model)
        # unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        # if len(missing_keys) > 0:
        #     print('Missing Keys: {}'.format(missing_keys))
        # if len(unexpected_keys) > 0:
        #     print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            # print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model
        # if not args.eval:
        #     test_stats, coco_evaluator = evaluate(
        #         model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        #     )
    if mode == "train" or mode == "finetune" or mode == "debug":
        print("Start training")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                sampler_train.set_epoch(epoch)
            for _, block in enumerate(model.backbone.backbone.blocks):
                if _ < 8:
                    for param in block.parameters():
                        param.requires_grad = True
            if mode == "finetune":
                for component in model.backbone:
                    for param in component.parameters():
                        param.requires_grad = False
                if version != "0":
                    for component in model.transformer.encoder:
                        for param in component.parameters():
                            param.requires_grad = False
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
            lr_scheduler.step()
            if args.output_dir:
                if mode == "finetune":
                    for component in model.backbone:
                        for param in component.parameters():
                            param.requires_grad = True
                    if version != "0":
                        for component in model.transformer.encoder:
                            for param in component.parameters():
                                param.requires_grad = True
                for _, block in enumerate(model.backbone.backbone.blocks):
                    if _ < 8:
                        for param in block.parameters():
                            param.requires_grad = False
                checkpoint_name = 'mDETD_'+args.version+'.pth'
                checkpoint_paths = [output_dir / checkpoint_name]
                # extra checkpoint before LR drop and every 1 epochs
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)

            # test_stats, coco_evaluator = evaluate(
            #     model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            # )

            # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            #              **{f'test_{k}': v for k, v in test_stats.items()},
            #              'epoch': epoch,
            #              'n_parameters': n_parameters}

            # if args.output_dir and utils.is_main_process():
            #     with (output_dir / "log.txt").open("a") as f:
            #         f.write(json.dumps(log_stats) + "\n")

            #     # for evaluation logs
            #     if coco_evaluator is not None:
            #         (output_dir / 'eval').mkdir(exist_ok=True)
            #         if "bbox" in coco_evaluator.coco_eval:
            #             filenames = ['latest.pth']
            #             if epoch % 1 == 0:
            #                 filenames.append(f'{epoch:03}.pth')
            #             for name in filenames:
            #                 torch.save(coco_evaluator.coco_eval["bbox"].eval,
            #                            output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    if mode == "val" or mode == "finetune" or mode == "debug" or mode == "finetune0" or mode == "finetune1" or mode == "finetune2":
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
    return        


if __name__ == '__main__':
    with open('../input.txt') as f:
        line = f.readline()
        mode, version = line.split()
    f.close()
    # mode, version = input().split()
    parser = argparse.ArgumentParser('ViT Deformable DETR training and evaluation script', parents=[get_args_parser(mode, version)])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('ViT Deformable DETR training and evaluation script', parents=[get_args_parser("debug", "2")])
#     args = parser.parse_args()
#     if args.output_dir:
#         Path(args.output_dir).mkdir(parents=True, exist_ok=True)
#     model, criterion, postprocessors = build_model(args)
#     pretrained_dict = torch.load('pre-trained checkpoints/full_coco_finetune.pth', map_location='cpu')
#     model_dict = model.state_dict()

#     # 1. filter out unnecessary keys
#     pretrained_dict = {k: v for k, v in pretrained_dict['model'].items() if k in model_dict}
#     # 2. overwrite entries in the existing state dict
#     model_dict.update(pretrained_dict) 
#     # 3. load the new state dict
#     model.load_state_dict(model_dict)
#     torch.save(model.state_dict(), 'pre-trained checkpoints/mDETD_2.pth')


    