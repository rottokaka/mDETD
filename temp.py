import os
import time
import argparse

import torch

from main import get_args_parser
from models import build_model
from datasets import build_dataset
from util.misc import nested_tensor_from_tensor_list


def get_benckmark_arg_parser():
    parser = argparse.ArgumentParser('Benchmark inference speed of Deformable DETR.')
    parser.add_argument('--num_iters', type=int, default=300, help='total iters to benchmark speed')
    parser.add_argument('--warm_iters', type=int, default=5, help='ignore first several iters that are very slow')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in inference')
    parser.add_argument('--resume', type=str, help='load the pre-trained checkpoint')
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ViT Deformable DETR training and evaluation script', parents=[get_args_parser("debug", "0")])
    args = parser.parse_args()
    model, _, _ = build_model(args)
    dataset = build_dataset('val', args)
    model.cuda()
    model.eval()
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt['model'])
    inputs = nested_tensor_from_tensor_list([dataset.__getitem__(0)[0].cuda() for _ in range(args.batch_size)])
    print("TTTTT")