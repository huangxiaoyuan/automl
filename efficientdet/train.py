#!/usr/bin/env python3
"""One-shot training script for EfficientDet on the 24-class bird dataset.

This wraps the repo's main.py / model_inspect.py with sensible defaults and
adds:
  - automatic num_examples_per_epoch (counts records in train TFRecords)
  - pre-flight checks (paths exist, hparams parses, num_classes is sane)
  - a `smoke` mode that runs ~2 steps to catch config bugs before a real run
  - optional post-train eval and SavedModel export

Run this from inside the cloned `automl/efficientdet/` directory so that
main.py and model_inspect.py are importable as subprocess targets.

Usage examples:
  # 1) smoke test first -- highly recommended
  python train_bird.py --action smoke

  # 2) real training (resumes from the COCO checkpoint)
  python train_bird.py --action train

  # 3) train, then eval, then export a SavedModel
  python train_bird.py --action all
"""

import argparse
import glob
import os
import subprocess
import sys

import tensorflow as tf
import yaml
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# --------------------------- EDIT THIS SECTION ------------------------------
# Paths are relative to where you run the script from. Absolute paths also OK.

TRAIN_PATTERN   = 'dataset/tfrecord/train-*.tfrecord'
VAL_PATTERN     = 'dataset/tfrecord/val-*.tfrecord'
VAL_JSON        = 'dataset/tfrecord/val.json'

MODEL_NAME      = 'efficientdet-d0'         # d0/d1/d2/d3/d4 ...
PRETRAINED_CKPT = 'efficientdet-d0'         # extracted COCO checkpoint dir
HPARAMS_FILE    = 'bird_config.yaml'        # the yaml we wrote earlier
MODEL_DIR       = 'traintmp/efficientdet_bird/' # checkpoints & TB logs
EXPORT_DIR      = 'savedmodel_bird/'        # where to put the SavedModel

# training hyper-params -- tune to your hardware / dataset
TRAIN_BATCH_SIZE = 4        # 4 for 8GB GPU, 8 for 12GB, 16 for 24GB
NUM_EPOCHS       = 50
EVAL_BATCH_SIZE  = 8
EXPECTED_NUM_CLASSES_INCL_BG = 25   # 24 birds + 1 background
# ----------------------------------------------------------------------------


def sh(cmd, dry=False):
  """Echo + run a shell-style command list. Raises on non-zero exit."""
  print('\n>>>', ' '.join(cmd), flush=True)
  if dry:
    return
  subprocess.run(cmd, check=True)


def count_tfrecord_examples(pattern):
  """Iterate the TFRecord(s) and count records. One pass; small/medium data."""
  files = sorted(tf.io.gfile.glob(pattern))
  if not files:
    raise FileNotFoundError('No files matched: %s' % pattern)
  n = 0
  for _ in tf.data.TFRecordDataset(files):
    n += 1
  return n, len(files)


def preflight():
  """Catch config errors before launching a long training run."""
  problems = []

  # 1. TFRecord files exist
  train_files = sorted(tf.io.gfile.glob(TRAIN_PATTERN))
  if not train_files:
    problems.append('No train TFRecords matched %r' % TRAIN_PATTERN)

  # 2. pretrained checkpoint dir exists (only matters if we're going to use it)
  if PRETRAINED_CKPT and not tf.io.gfile.exists(PRETRAINED_CKPT):
    problems.append('Pretrained ckpt path not found: %s' % PRETRAINED_CKPT)

  # 3. hparams file parses and num_classes is the EXPECTED value
  if not tf.io.gfile.exists(HPARAMS_FILE):
    problems.append('hparams file missing: %s' % HPARAMS_FILE)
  else:
    with open(HPARAMS_FILE) as f:
      cfg = yaml.safe_load(f) or {}
    nc = cfg.get('num_classes')
    if nc != EXPECTED_NUM_CLASSES_INCL_BG:
      problems.append(
          'hparams num_classes=%r but expected %d (24 real + 1 background). '
          'Forgetting the +1 is the #1 cause of silent training failures.'
          % (nc, EXPECTED_NUM_CLASSES_INCL_BG))
    lm = cfg.get('label_map') or {}
    if lm:
      keys = sorted(lm.keys())
      if keys[0] != 1:
        problems.append('label_map keys should start at 1 (0 = background); '
                        'got smallest key = %r' % keys[0])
      if len(keys) != EXPECTED_NUM_CLASSES_INCL_BG - 1:
        problems.append('label_map has %d entries; expected %d.'
                        % (len(keys), EXPECTED_NUM_CLASSES_INCL_BG - 1))

  if problems:
    print('\nPre-flight check FAILED:')
    for p in problems:
      print('  -', p)
    sys.exit(2)
  print('Pre-flight check passed.')


def compute_examples_per_epoch():
  n, nfiles = count_tfrecord_examples(TRAIN_PATTERN)
  steps = n // TRAIN_BATCH_SIZE
  print('Train set: %d examples across %d shards -> %d steps/epoch (bs=%d)'
        % (n, nfiles, steps, TRAIN_BATCH_SIZE))
  return n


def cmd_train(num_examples, smoke=False):
  cmd = [
      'python', 'main.py',
      '--mode=train',
      '--train_file_pattern=%s' % TRAIN_PATTERN,
      '--model_name=%s' % MODEL_NAME,
      '--model_dir=%s' % (MODEL_DIR + ('_smoke' if smoke else '')),
      '--train_batch_size=%d' % (2 if smoke else TRAIN_BATCH_SIZE),
      '--num_examples_per_epoch=%d' % (min(num_examples, 16) if smoke
                                       else num_examples),
      '--num_epochs=%d' % (1 if smoke else NUM_EPOCHS),
      '--hparams=%s' % HPARAMS_FILE,
  ]
  if PRETRAINED_CKPT:
    cmd.append('--ckpt=%s' % PRETRAINED_CKPT)
  return cmd


def cmd_eval():
  return [
      'python', 'main.py',
      '--mode=eval',
      '--val_file_pattern=%s' % VAL_PATTERN,
      '--val_json_file=%s' % VAL_JSON,
      '--model_name=%s' % MODEL_NAME,
      '--model_dir=%s' % MODEL_DIR,
      '--eval_batch_size=%d' % EVAL_BATCH_SIZE,
      '--hparams=%s' % HPARAMS_FILE,
  ]


def cmd_export():
  return [
      'python', 'model_inspect.py',
      '--runmode=saved_model',
      '--model_name=%s' % MODEL_NAME,
      '--ckpt_path=%s' % MODEL_DIR,
      '--saved_model_dir=%s' % EXPORT_DIR,
      '--hparams=%s' % HPARAMS_FILE,
  ]


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--action', default='train',
                 choices=['smoke', 'train', 'eval', 'export', 'all'],
                 help='smoke = ~2 steps to verify pipeline; '
                      'train = full training; '
                      'eval = run COCO eval on val set; '
                      'export = export SavedModel from MODEL_DIR; '
                      'all = train -> eval -> export.')
  p.add_argument('--dry_run', action='store_true',
                 help='Print the commands but do not execute.')
  args = p.parse_args()

  preflight()
  num_examples = compute_examples_per_epoch()

  if args.action == 'smoke':
    sh(cmd_train(num_examples, smoke=True), dry=args.dry_run)
    print('\nSmoke test finished. If you saw loss values, pipeline is OK.')
    return

  if args.action in ('train', 'all'):
    sh(cmd_train(num_examples, smoke=False), dry=args.dry_run)

  if args.action in ('eval', 'all'):
    if not tf.io.gfile.glob(VAL_PATTERN):
      print('Skipping eval: no val TFRecords at %r' % VAL_PATTERN)
    else:
      sh(cmd_eval(), dry=args.dry_run)

  if args.action in ('export', 'all'):
    sh(cmd_export(), dry=args.dry_run)
    print('\nSavedModel written to:', EXPORT_DIR)


if __name__ == '__main__':
  main()
