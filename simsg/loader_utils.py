from simsg.data.vg import SceneGraphNoPairsDataset, collate_fn_nopairs
from simsg.data.clevr import SceneGraphWithPairsDataset, collate_fn_withpairs
from simsg.data.robot import RobotDataset

import json
from torch.utils.data import DataLoader

def build_robot_supervised_train_dsets(args):
  print("building fully supervised %s dataset" % args.dataset)
  with open(args.vocab_json, 'r') as f:
    vocab = json.load(f)
  dset_kwargs = {
    'image_dir': args.train_image_dir,
    'instances_json': args.train_instances_json,
    'src_image_dir': args.train_src_image_dir,
    'src_instances_json': args.train_src_instances_json,
    'image_size': args.image_size
  }
  train_dset = RobotDataset(**dset_kwargs)
  iter_per_epoch = len(train_dset) // args.batch_size
  print('There are %d iterations per epoch' % iter_per_epoch)

  dset_kwargs['image_dir'] = args.val_image_dir
  dset_kwargs['instances_json'] = args.val_instances_json
  dset_kwargs['src_image_dir'] = args.val_src_image_dir
  dset_kwargs['src_instances_json'] = args.val_src_instances_json
  val_dset = RobotDataset(**dset_kwargs)

  dset_kwargs['image_dir'] = args.test_image_dir
  dset_kwargs['instances_json'] = args.test_instances_json
  dset_kwargs['src_image_dir'] = args.test_src_image_dir
  dset_kwargs['src_instances_json'] = args.test_src_instances_json
  test_dset = RobotDataset(**dset_kwargs)

  return vocab, train_dset, val_dset, test_dset

def build_clevr_supervised_train_dsets(args):
  print("building fully supervised %s dataset" % args.dataset)
  with open(args.vocab_json, 'r') as f:
    vocab = json.load(f)
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.train_h5,
    'image_dir': args.vg_image_dir,
    'image_size': args.image_size,
    'max_samples': args.num_train_samples,
    'max_objects': args.max_objects_per_image,
    'use_orphaned_objects': args.vg_use_orphaned_objects,
    'include_relationships': args.include_relationships,
  }
  train_dset = SceneGraphWithPairsDataset(**dset_kwargs)
  iter_per_epoch = len(train_dset) // args.batch_size
  print('There are %d iterations per epoch' % iter_per_epoch)

  dset_kwargs['h5_path'] = args.val_h5
  del dset_kwargs['max_samples']
  val_dset = SceneGraphWithPairsDataset(**dset_kwargs)

  dset_kwargs['h5_path'] = args.test_h5
  test_dset = SceneGraphWithPairsDataset(**dset_kwargs)

  return vocab, train_dset, val_dset, test_dset


def build_dset_nopairs(args, checkpoint):

  vocab = checkpoint['model_kwargs']['vocab']
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.data_h5,
    'image_dir': args.data_image_dir,
    'image_size': args.image_size,
    'max_objects': checkpoint['args']['max_objects_per_image'],
    'use_orphaned_objects': checkpoint['args']['vg_use_orphaned_objects'],
    'mode': args.mode,
    'predgraphs': args.predgraphs
  }
  dset = SceneGraphNoPairsDataset(**dset_kwargs)

  return dset


def build_dset_withpairs(args, checkpoint, vocab_t):

  vocab = vocab_t
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.data_h5,
    'image_dir': args.data_image_dir,
    'image_size': args.image_size,
    'max_objects': checkpoint['args']['max_objects_per_image'],
    'use_orphaned_objects': checkpoint['args']['vg_use_orphaned_objects'],
    'mode': args.mode
  }
  dset = SceneGraphWithPairsDataset(**dset_kwargs)

  return dset


def build_eval_loader(args, checkpoint, vocab_t=None, no_gt=False):

  if args.dataset == 'vg' or (no_gt and args.dataset == 'clevr'):
    dset = build_dset_nopairs(args, checkpoint)
    collate_fn = collate_fn_nopairs
  elif args.dataset == 'clevr':
    dset = build_dset_withpairs(args, checkpoint, vocab_t)
    collate_fn = collate_fn_withpairs

  loader_kwargs = {
    'batch_size': 1,
    'num_workers': args.loader_num_workers,
    'shuffle': args.shuffle,
    'collate_fn': collate_fn,
  }
  loader = DataLoader(dset, **loader_kwargs)

  return loader


def build_train_dsets(args):
  print("building unpaired %s dataset" % args.dataset)
  with open(args.vocab_json, 'r') as f:
    vocab = json.load(f)
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.train_h5,
    'image_dir': args.vg_image_dir,
    'image_size': args.image_size,
    'max_samples': args.num_train_samples,
    'max_objects': args.max_objects_per_image,
    'use_orphaned_objects': args.vg_use_orphaned_objects,
    'include_relationships': args.include_relationships,
  }
  train_dset = SceneGraphNoPairsDataset(**dset_kwargs)
  iter_per_epoch = len(train_dset) // args.batch_size
  print('There are %d iterations per epoch' % iter_per_epoch)

  dset_kwargs['h5_path'] = args.val_h5
  del dset_kwargs['max_samples']
  val_dset = SceneGraphNoPairsDataset(**dset_kwargs)

  return vocab, train_dset, val_dset


def build_train_loaders(args, shuffle_train=True):

  print(args.dataset)
  if args.dataset == 'vg' or (args.dataset == "clevr" and not args.is_supervised):
    vocab, train_dset, val_dset = build_train_dsets(args)
    collate_fn = collate_fn_nopairs
  elif args.dataset == 'clevr':
    vocab, train_dset, val_dset, test_dset = build_clevr_supervised_train_dsets(args)
    collate_fn = collate_fn_withpairs
  elif args.dataset == 'robot':
    vocab, train_dset, val_dset, test_dset = build_robot_supervised_train_dsets(args)
    collate_fn = collate_fn_withpairs

  loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': shuffle_train,
    'collate_fn': collate_fn,
  }
  train_loader = DataLoader(train_dset, **loader_kwargs)

  loader_kwargs['shuffle'] = args.shuffle_val
  val_loader = DataLoader(val_dset, **loader_kwargs)

  return vocab, train_loader, val_loader
