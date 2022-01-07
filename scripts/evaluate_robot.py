#!/usr/bin/python
#
# Copyright 2020 Azade Farshad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script can be used to test changes for clevr
"""

import argparse, json
import os

import torch
from collections import Counter

from imageio import imsave
import matplotlib.pyplot as plt

from simsg.data import imagenet_deprocess_batch
from simsg.model import SIMSGModel
from simsg.utils import int_tuple, bool_flag

import pytorch_ssim
from simsg.metrics import jaccard
#perceptual error
from PerceptualSimilarity import models

import cv2
import numpy as np

from simsg.loader_utils import build_train_loaders
from scripts.eval_utils import bbox_coordinates_with_margin, makedir, query_image_by_semantic_id, save_graph_json

DATA_DIR = os.path.expanduser('~/projects/simsg/data/robot_supervised')

parser = argparse.ArgumentParser()

# For robot dataset
parser.add_argument('--train_image_dir', default=os.path.join(DATA_DIR, 'train1/train_images'))
parser.add_argument('--train_instances_json', default=os.path.join(DATA_DIR, 'train1/instances_train.json'))
parser.add_argument('--train_src_image_dir', default=os.path.join(DATA_DIR, 'train0/train_images'))
parser.add_argument('--train_src_instances_json', default=os.path.join(DATA_DIR, 'train0/instances_train.json'))
parser.add_argument('--val_image_dir', default=os.path.join(DATA_DIR, 'val1/val_images'))
parser.add_argument('--val_instances_json', default=os.path.join(DATA_DIR, 'val1/instances_val.json'))
parser.add_argument('--val_src_image_dir', default=os.path.join(DATA_DIR, 'val0/val_images'))
parser.add_argument('--val_src_instances_json', default=os.path.join(DATA_DIR, 'val0/instances_val.json'))
parser.add_argument('--test_image_dir', default=os.path.join(DATA_DIR, 'val1/val_images'))
parser.add_argument('--test_instances_json', default=os.path.join(DATA_DIR, 'val1/instances_val.json'))
parser.add_argument('--test_src_image_dir', default=os.path.join(DATA_DIR, 'val0/val_images'))
parser.add_argument('--test_src_instances_json', default=os.path.join(DATA_DIR, 'val0/instances_val.json'))

parser.add_argument('--exp_dir', default='./experiments/robot/')
parser.add_argument('--experiment', default="spade_robot", type=str)
parser.add_argument('--checkpoint', default='experiments/robot/spade_robot_model.pt')
parser.add_argument('--image_size', default=(96, 128), type=int_tuple)
parser.add_argument('--loader_num_workers', default=0, type=int)

parser.add_argument('--vocab_json', default=os.path.join(DATA_DIR, 'vocab.json'))

args = parser.parse_args()
args.dataset = "robot"
args.batch_size = 16
args.shuffle_val = False


def build_model(args, checkpoint):
  model = SIMSGModel(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'])
  model.eval()
  model.image_size = args.image_size
  model.cuda()
  return model


def run_model(args, checkpoint, loader=None):
  output_dir = args.exp_dir
  img_dir = os.path.join(output_dir, 'train_images')
  if not os.path.exists(img_dir):
    os.makedirs(img_dir)

  model = build_model(args, checkpoint)
  vocab, train_loader, val_loader = build_train_loaders(args, False)

  img_index = 0
  for b_index, batch in enumerate(train_loader):
    imgs, imgs_src, objs, objs_src, boxes, boxes_src, triples, triples_src, obj_to_img, triple_to_img, imgs_in = [x.cuda() for x in batch]

    model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=None, src_image=imgs_in, imgs_src=imgs_src)

    imgs_pred, masks_pred, noised_srcs, _ = model_out

    for i in range(len(imgs)):
      combined_img = torch.cat((noised_srcs[i,:3,:,:], imgs[i], imgs_pred[i]), dim=2)
      img = combined_img.cpu().detach().numpy().transpose(1, 2, 0)*255
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      cv2.imwrite(os.path.join(img_dir, f'{img_index}.png'), img)
      img_index += 1
      
    print(img_index)


def main(args):

  got_checkpoint = args.checkpoint is not None

  if got_checkpoint:
    checkpoint = torch.load(args.checkpoint)
    print('Loading model from ', args.checkpoint)
    run_model(args, checkpoint)
  else:
    print('--checkpoint not specified')


if __name__ == '__main__':
  main(args)
