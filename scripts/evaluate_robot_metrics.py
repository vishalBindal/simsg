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

print_every = 1

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

parser.add_argument('--with_query_image', default=False, type=bool)

args = parser.parse_args()
args.dataset = "robot"
args.batch_size = 16
args.shuffle_val = False
output_file = os.path.join(args.exp_dir ,args.experiment + ".txt")

def build_model(args, checkpoint):
  model = SIMSGModel(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'])
  model.eval()
  model.image_size = args.image_size
  model.cuda()
  return model

def tripleToObjID(triples, objs):
  triples_new = []
  for [s,p,o] in triples:
    s2 = int(objs[s].cpu())
    o2 = int(objs[o].cpu())
    triples_new.append([s2, int(p.cpu()), o2])
  return triples_new


def get_triples_names(triples, vocab):
  new_triples = []
  triples = list(triples)
  for i in range(len(triples)):
    s, p, o = triples[i]
    new_triples.append([vocab['object_idx_to_name'][s], vocab['pred_idx_to_name'][p], vocab['object_idx_to_name'][o]])
  return new_triples


def get_def_dict():
  new_dict = {}
  new_dict['replace'] = []
  new_dict['reposition'] = []
  new_dict['remove'] = []
  new_dict['addition'] = []
  return new_dict


def calculate_scores(mae_per_image, mae_roi_per_image, ssim_per_image, ssim_rois,
                       perceptual_image, perceptual_roi):
    mae_all = np.mean(np.hstack(mae_per_image), dtype=np.float64)
    mae_std = np.std(np.hstack(mae_per_image), dtype=np.float64)
    mae_roi = np.mean(mae_roi_per_image, dtype=np.float64)
    mae_roi_std = np.std(mae_roi_per_image, dtype=np.float64)
    # ssim_all = np.mean(ssim_per_image, dtype=np.float64)
    # ssim_std = np.std(ssim_per_image, dtype=np.float64)
    # ssim_roi = np.mean(ssim_rois, dtype=np.float64)
    # ssim_roi_std = np.std(ssim_rois, dtype=np.float64)
    # percept error -----------
    percept_all = np.mean(perceptual_image, dtype=np.float64)
    # print(perceptual_image, percept_all)
    percept_all_std = np.std(perceptual_image, dtype=np.float64)
    percept_roi = np.mean(perceptual_roi, dtype=np.float64)
    percept_roi_std = np.std(perceptual_roi, dtype=np.float64)
    # ------------------------

    print()
    print('MAE: Mean {:.6f}, Std {:.6f}'.format(mae_all, mae_std))
    print('MAE-RoI: Mean {:.6f}, Std {:.6f}'.format(mae_roi, mae_roi_std))
    # print('SSIM: Mean {:.6f}, Std {:.6f}'.format(ssim_all, ssim_std))
    # print('SSIM-RoI: Mean {:.6f}, Std {:.6f}'.format(ssim_roi, ssim_roi_std))
    print('LPIPS: Mean {:.6f}, Std {:.6f}'.format(percept_all, percept_all_std))
    print('LPIPS-RoI: Mean {:.6f}, Std {:.6f}'.format(percept_roi, percept_roi_std))

    with open(output_file, "a+") as f:
        f.write("Mean All\n")
        f.write('MAE: Mean {:.6f}, Std {:.6f}\n'.format(mae_all, mae_std))
        f.write('MAE-RoI: Mean {:.6f}, Std {:.6f}\n'.format(mae_roi, mae_roi_std))
        # f.write('SSIM: Mean {:.6f}, Std {:.6f}\n'.format(ssim_all, ssim_std))
        # f.write('SSIM-RoI: Mean {:.6f}, Std {:.6f}\n'.format(ssim_roi, ssim_roi_std))
        f.write('LPIPS: Mean {:.6f}, Std {:.6f}\n'.format(percept_all, percept_all_std))
        f.write('LPIPS-RoI: Mean {:.6f}, Std {:.6f}\n'.format(percept_roi, percept_roi_std))


modes = ['reposition']


def run_model(args, checkpoint, loader=None):
  output_dir = args.exp_dir
  img_dir = os.path.join(output_dir, 'val_images')
  if not os.path.exists(img_dir):
    os.makedirs(img_dir)

  f = open(output_dir + "/result_ids.txt", "w")

  model = build_model(args, checkpoint)
  vocab, train_loader, val_loader = build_train_loaders(args, False)

  
  img_idx = 0
  mae_per_image_all = []
  mae_per_image = get_def_dict()
  mae_roi_per_image_all = []
  mae_roi_per_image = get_def_dict()
  # ssim_per_image_all = []
  # ssim_per_image = get_def_dict()
  # ssim_rois_all = []
  # ssim_rois = get_def_dict()
  margin = 2

  ## Initializing the perceptual loss model
  lpips_model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True)
  perceptual_error_image_all = []
  perceptual_error_image = get_def_dict()
  perceptual_error_roi_all = []
  perceptual_error_roi = get_def_dict()

  img_index = 0
  for b_index, batch in enumerate(train_loader):
    imgs, imgs_src, objs, objs_src, boxes, boxes_src, triples, triples_src, obj_to_img, triple_to_img, imgs_in = [x.cuda() for x in batch]
    imgs_gt = imgs_src
    imgs_target_gt = imgs

    graph_set_bef = Counter(tuple(row) for row in tripleToObjID(triples_src, objs_src))
    obj_set_bef = Counter([int(obj.cpu()) for obj in objs_src])
    graph_set_aft = Counter(tuple(row) for row in tripleToObjID(triples, objs))
    obj_set_aft = Counter([int(obj.cpu()) for obj in objs])

    mode = "reposition"
    changes = (graph_set_bef - graph_set_aft) + (graph_set_aft - graph_set_bef)
    idx_cnt = np.zeros((25,1))
    for [s,p,o] in list(changes):
      idx_cnt[s] += 1
      idx_cnt[o] += 1

    obj_ids = idx_cnt.argmax(0)
    id_src = (objs_src == obj_ids[0]).nonzero()
    box_src = boxes_src[id_src[0]]
    new_ids = (objs == obj_ids[0]).nonzero()
    # boxes[new_ids[0]] = box_src

    new_ids = [int(new_id.cpu()) for new_id in new_ids]

    query_feats = None

    img_gt_filename = '%04d_gt_src.png' % (img_idx)
    img_target_gt_filename = '%04d_gt_target.png' % (img_idx)
    img_pred_filename = '%04d_changed.png' % (img_idx)
    img_filename_noised = '%04d_noised.png' % (img_idx)

    triples_ = triples

    boxes_gt = boxes

    keep_box_idx = torch.ones_like(objs.unsqueeze(1), dtype=torch.float)
    keep_feat_idx = torch.ones_like(objs.unsqueeze(1), dtype=torch.float)
    keep_image_idx = torch.ones_like(objs.unsqueeze(1), dtype=torch.float)

    subject_node = new_ids[0]
    keep_image_idx[subject_node] = 0

    keep_box_idx[subject_node] = 0

    model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=None, src_image=imgs_in, imgs_src=imgs_src)

    imgs_pred, masks_pred, noised_srcs, _ = model_out

    # MAE per image
    curr_mae = torch.mean(
      torch.abs(imgs - imgs_pred).view(imgs.shape[0], -1), 1).cpu().detach().numpy()
    mae_per_image[mode].append(curr_mae)
    mae_per_image_all.append(curr_mae)

    for s in range(imgs.shape[0]):
      curr_mae_roi = 0
      num = 0
      for bi in range(boxes.shape[0]):
        if obj_to_img[bi] != s:
          continue
        # get coordinates of target
        left, right, top, bottom = bbox_coordinates_with_margin(boxes[bi], margin, imgs)
        if left > right or top > bottom:
          continue
        # print("bboxes with margin: ", left, right, top, bottom)

        # calculate errors only in RoI one by one
        curr_mae_roi += torch.sum(
          torch.abs(imgs[s, :, top:bottom, left:right] - imgs_pred[s, :, top:bottom, left:right])).cpu().item()
        num += (bottom-top)*(right-left)
      curr_mae_roi /= num
      mae_roi_per_image_all.append(curr_mae_roi)

      # curr_ssim = pytorch_ssim.ssim((imgs[s:s + 1, :, :, :] / 255.0).cuda(),
                          # (imgs_pred[s:s + 1, :, :, :] / 255.0).cuda(), window_size=3).cpu().item()
      # ssim_per_image_all.append(curr_ssim)
      # ssim_per_image[mode].append(curr_ssim)

      # curr_ssim_roi = pytorch_ssim.ssim(imgs[s:s + 1, :, top:bottom, left:right] / 255.0,
                          # imgs_pred[s:s + 1, :, top:bottom, left:right] / 255.0, window_size=3).cpu().item()
      # ssim_rois_all.append(curr_ssim_roi)
      # ssim_rois[mode].append(curr_ssim_roi)

      # imgs_pred_norm = imgs_pred[s:s + 1, :, :, :] / 127.5 - 1
      # imgs_gt_norm = imgs[s:s + 1, :, :, :] / 127.5 - 1

      # curr_lpips = lpips_model.forward(imgs_pred_norm, imgs_gt_norm).detach().cpu().numpy()
      # perceptual_error_image_all.append(curr_lpips)
      # perceptual_error_image[mode].append(curr_lpips)

    img_idx += 1

    ssim_per_image_all = None
    ssim_rois_all = None
    ssim_per_image = None
    ssim_rois = None

    if img_idx % print_every == 0:
      calculate_scores(mae_per_image_all, mae_roi_per_image_all, ssim_per_image_all,
                       ssim_rois_all, perceptual_error_image_all, perceptual_error_roi_all)
  f.close()


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
