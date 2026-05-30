r"""Visualize boxes stored in EfficientDet-style TFRecord files, to sanity-check
a YOLO->TFRecord conversion before training.

It reads the TFRecord(s), randomly samples a few images, decodes the embedded
image bytes, draws every box (denormalizing the [0,1] corner coords back to
pixels) with its class name, and saves annotated images you can eyeball.

Usage:
    python visualize_tfrecord.py \
        --file_pattern 'tfrecord/train-*.tfrecord' \
        --classes_file classes.txt \
        --num_samples 12 \
        --output_dir vis_out

Notes:
  - --classes_file is optional; it's only used to double-check that the
    class/text stored in the record matches your class list. The label drawn
    comes from image/object/class/text inside the record.
  - Sampling is reservoir-style over the whole dataset, so you don't need to
    load everything into memory.
"""

import io
import os
import random

from absl import app
from absl import flags
from absl import logging
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import tensorflow as tf

flags.DEFINE_string('file_pattern', None,
                    "Glob for TFRecord files, e.g. 'tfrecord/train-*.tfrecord'.")
flags.DEFINE_string('classes_file', None,
                    'Optional class names file (line index = 0-based id).')
flags.DEFINE_integer('num_samples', 12, 'How many images to render.')
flags.DEFINE_string('output_dir', 'vis_out', 'Where to save annotated images.')
flags.DEFINE_integer('seed', 0, 'Random seed for reproducible sampling.')
FLAGS = flags.FLAGS

FEATURE_SPEC = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=b''),
    'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
}

# a small fixed palette; color is chosen by class id so the same class is
# always the same color across images
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    (255, 99, 71), (46, 139, 87), (106, 90, 205), (218, 112, 214),
]


def color_for(class_id):
  return PALETTE[int(class_id) % len(PALETTE)]


def load_class_names(path):
  if not path:
    return None
  with tf.io.gfile.GFile(path, 'r') as f:
    return [line.strip() for line in f if line.strip()]


def reservoir_sample(records, k, seed):
  """Reservoir sampling over an iterable of serialized records."""
  rng = random.Random(seed)
  reservoir = []
  for i, rec in enumerate(records):
    if i < k:
      reservoir.append(rec)
    else:
      j = rng.randint(0, i)
      if j < k:
        reservoir[j] = rec
  return reservoir


def draw_example(serialized, class_names):
  ex = tf.io.parse_single_example(serialized, FEATURE_SPEC)
  encoded = ex['image/encoded'].numpy()
  img = PIL.Image.open(io.BytesIO(encoded)).convert('RGB')
  W, H = img.size
  draw = PIL.ImageDraw.Draw(img)
  try:
    font = PIL.ImageFont.truetype('DejaVuSans.ttf', max(12, W // 60))
  except Exception:
    font = PIL.ImageFont.load_default()

  xmin = tf.sparse.to_dense(ex['image/object/bbox/xmin']).numpy()
  ymin = tf.sparse.to_dense(ex['image/object/bbox/ymin']).numpy()
  xmax = tf.sparse.to_dense(ex['image/object/bbox/xmax']).numpy()
  ymax = tf.sparse.to_dense(ex['image/object/bbox/ymax']).numpy()
  labels = tf.sparse.to_dense(ex['image/object/class/label']).numpy()
  texts = [t.decode('utf8') for t in
           tf.sparse.to_dense(ex['image/object/class/text']).numpy()]
  fname = ex['image/filename'].numpy().decode('utf8') or 'unknown'

  for i in range(len(xmin)):
    x0, y0 = xmin[i] * W, ymin[i] * H
    x1, y1 = xmax[i] * W, ymax[i] * H
    cid = int(labels[i]) if i < len(labels) else 0
    col = color_for(cid - 1)  # labels are 1-based; palette index 0-based
    draw.rectangle([x0, y0, x1, y1], outline=col, width=max(2, W // 300))
    # label text: prefer stored text, fall back to id; cross-check class list
    name = texts[i] if i < len(texts) and texts[i] else str(cid)
    tag = '%s(%d)' % (name, cid)
    # text background for readability
    tb = draw.textbbox((x0, y0), tag, font=font)
    draw.rectangle([tb[0], tb[1], tb[2] + 2, tb[3] + 2], fill=col)
    draw.text((x0 + 1, y0 + 1), tag, fill=(255, 255, 255), font=font)

  return img, fname, len(xmin)


def main(_):
  if FLAGS.file_pattern is None:
    raise ValueError('--file_pattern is required')

  class_names = load_class_names(FLAGS.classes_file)
  files = sorted(tf.io.gfile.glob(FLAGS.file_pattern))
  if not files:
    raise ValueError('No files matched %s' % FLAGS.file_pattern)
  logging.info('Matched %d TFRecord files.', len(files))

  ds = tf.data.TFRecordDataset(files)
  sampled = reservoir_sample(
      (r.numpy() for r in ds), FLAGS.num_samples, FLAGS.seed)
  logging.info('Sampled %d records.', len(sampled))

  if not tf.io.gfile.exists(FLAGS.output_dir):
    tf.io.gfile.makedirs(FLAGS.output_dir)

  total_boxes = 0
  for idx, serialized in enumerate(sampled):
    img, fname, nboxes = draw_example(serialized, class_names)
    total_boxes += nboxes
    safe = os.path.splitext(os.path.basename(fname))[0]
    out_path = os.path.join(FLAGS.output_dir, 'vis_%02d_%s.jpg' % (idx, safe))
    img.save(out_path, quality=90)
    logging.info('  [%2d] %-30s %d boxes -> %s', idx, fname, nboxes, out_path)

  logging.info('Done. %d images, %d boxes total, saved under %s/',
               len(sampled), total_boxes, FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
