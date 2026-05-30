r"""Convert a YOLO-format detection dataset (organized in per-class subfolders)
to TFRecord files for EfficientDet (google/automl).

Expected dataset layout (the dataset has ALREADY been split):
    train/
      <class_folder_1>/
        images/ *.jpg|*.png|...
        labels/ *.txt           (same stem as the image)
      <class_folder_2>/
        images/ ...
        labels/ ...
      ... (24 class folders, NOT counting background)
    val/   (same structure)
    test/  (same structure)

IMPORTANT about class ids:
  The class_id at the start of each YOLO label line is the GLOBAL id in [0, 23].
  The per-class folder is just an organizational convenience -- it is NOT used
  to decide the category. We trust the id inside each .txt file.

YOLO label line:  class_id x_center y_center width height   (all normalized,
center+size, 0-based class id).

EfficientDet TFRecord feature keys (per the repo's create_coco_tfrecord.py):
    image/encoded, image/format, image/filename, image/source_id,
    image/height, image/width,
    image/object/bbox/{xmin,xmax,ymin,ymax}  -> NORMALIZED corner coords [0,1]
    image/object/class/label  -> int64, 1-based ids  (YOLO id + 1)
    image/object/class/text   -> class name bytes
    image/object/area, image/object/is_crowd

Usage:
    python yolo_nested_to_tfrecord.py \
        --split_dir /path/to/train \
        --classes_file classes.txt \
        --output_prefix tfrecord/train \
        --num_shards 32 \
        --json_output tfrecord/train.json
python yolo_nested_to_tfrecord_v2.py   --split_dir /data/datesets/birds/Macao_Ebird_24_detect_V2_train/train  --classes_file classes.txt   --output_prefix tfrecord/train
  --num_shards 32  --json_output tfrecord/train.json
python visualize_tfrecord.py --file_pattern 'tfrecord/train-00001-of-00032.tfrecord' --classes_file classes.txt --num_samples 12
  --output_dir vis_out

classes.txt = one class name per line; line index = global class id (0-based),
so it must have exactly 24 lines for your dataset.
"""

import hashlib
import io
import json
import os

from absl import app
from absl import flags
from absl import logging
import PIL.Image
import tensorflow as tf

flags.DEFINE_string('split_dir', None,
                    'Root dir of one split, e.g. .../train (contains per-class '
                    'subfolders, each with images/ and labels/).')
flags.DEFINE_string('classes_file', None,
                    'Text file: one class name per line; line index = global '
                    '0-based class id. Must have 24 lines for your dataset.')
flags.DEFINE_string('output_prefix', None,
                    'Output path prefix, e.g. tfrecord/train.')
flags.DEFINE_integer('num_shards', 32, 'Number of output TFRecord shards.')
flags.DEFINE_string('json_output', None,
                    'Optional COCO-style json (use as --val_json_file at eval).')
FLAGS = flags.FLAGS

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def load_class_names(path):
  with tf.io.gfile.GFile(path, 'r') as f:
    names = [line.strip() for line in f if line.strip()]
  return names


def collect_pairs(split_dir):
  """Walk per-class subfolders and return (image_path, label_path) pairs.

  For each class folder, match labels/<stem>.txt to images/<stem>.<ext>.
  Images without a label file are kept (treated as having zero objects);
  labels without an image are skipped with a warning.
  """
  pairs = []
  class_folders = sorted(
      d for d in tf.io.gfile.listdir(split_dir)
      if tf.io.gfile.isdir(os.path.join(split_dir, d.rstrip('/'))))
  for cf in class_folders:
    cf = cf.rstrip('/')
    img_dir = os.path.join(split_dir, cf, 'images')
    lbl_dir = os.path.join(split_dir, cf, 'labels')
    if not tf.io.gfile.exists(img_dir):
      logging.warning('No images/ in folder %s, skipping.', cf)
      continue
    # index labels by stem for quick lookup
    label_by_stem = {}
    if tf.io.gfile.exists(lbl_dir):
      for lf in tf.io.gfile.listdir(lbl_dir):
        if lf.endswith('.txt'):
          label_by_stem[os.path.splitext(lf)[0]] = os.path.join(lbl_dir, lf)
    for imf in tf.io.gfile.listdir(img_dir):
      if not imf.endswith(IMG_EXTS):
        continue
      stem = os.path.splitext(imf)[0]
      image_path = os.path.join(img_dir, imf)
      label_path = label_by_stem.get(stem)  # may be None -> no objects
      pairs.append((image_path, label_path))
  return pairs


def yolo_line_to_corners(parts):
  cls = int(float(parts[0]))
  xc, yc, w, h = (float(p) for p in parts[1:5])
  xmin = max(0.0, xc - w / 2.0)
  ymin = max(0.0, yc - h / 2.0)
  xmax = min(1.0, xc + w / 2.0)
  ymax = min(1.0, yc + h / 2.0)
  return cls, xmin, ymin, xmax, ymax


def create_tf_example(image_path, label_path, class_names, image_id, id_counts):
  with tf.io.gfile.GFile(image_path, 'rb') as f:
    encoded = f.read()
  image = PIL.Image.open(io.BytesIO(encoded))
  width, height = image.size
  image_format = b'png' if image.format == 'PNG' else b'jpeg'
  key = hashlib.sha256(encoded).hexdigest()

  xmins, xmaxs, ymins, ymaxs = [], [], [], []
  class_ids, class_text, areas, is_crowd = [], [], [], []
  coco_annotations = []

  if label_path and tf.io.gfile.exists(label_path):
    with tf.io.gfile.GFile(label_path, 'r') as f:
      for line in f:
        parts = line.split()
        if len(parts) < 5:
          continue
        cls, xmin, ymin, xmax, ymax = yolo_line_to_corners(parts)
        if xmax <= xmin or ymax <= ymin:
          continue
        if cls < 0 or cls >= len(class_names):
          logging.warning('class id %d out of range in %s, skipping box.',
                           cls, label_path)
          continue
        id_counts[cls] = id_counts.get(cls, 0) + 1
        xmins.append(xmin); ymins.append(ymin)
        xmaxs.append(xmax); ymaxs.append(ymax)
        class_ids.append(cls + 1)                    # 0-based -> 1-based
        class_text.append(class_names[cls].encode('utf8'))
        areas.append((xmax - xmin) * (ymax - ymin) * width * height)
        is_crowd.append(0)
        coco_annotations.append({
            'image_id': image_id, 'category_id': cls + 1,
            'bbox': [xmin * width, ymin * height,
                     (xmax - xmin) * width, (ymax - ymin) * height],
            'area': (xmax - xmin) * (ymax - ymin) * width * height,
            'iscrowd': 0,
        })

  feature_dict = {
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/filename': _bytes_feature(os.path.basename(image_path).encode('utf8')),
      'image/source_id': _bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256': _bytes_feature(key.encode('utf8')),
      'image/encoded': _bytes_feature(encoded),
      'image/format': _bytes_feature(image_format),
      'image/object/bbox/xmin': _float_list_feature(xmins),
      'image/object/bbox/xmax': _float_list_feature(xmaxs),
      'image/object/bbox/ymin': _float_list_feature(ymins),
      'image/object/bbox/ymax': _float_list_feature(ymaxs),
      'image/object/class/text': _bytes_list_feature(class_text),
      'image/object/class/label': _int64_list_feature(class_ids),
      'image/object/area': _float_list_feature(areas),
      'image/object/is_crowd': _int64_list_feature(is_crowd),
  }
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example, coco_annotations, (width, height)


def main(_):
  for flag in ('split_dir', 'classes_file', 'output_prefix'):
    if getattr(FLAGS, flag) is None:
      raise ValueError('--%s is required' % flag)

  class_names = load_class_names(FLAGS.classes_file)
  logging.info('Loaded %d class names.', len(class_names))

  pairs = collect_pairs(FLAGS.split_dir)
  logging.info('Collected %d image/label pairs across class folders.', len(pairs))
  if not pairs:
    raise ValueError('No images found. Check --split_dir structure.')

  out_dir = os.path.dirname(FLAGS.output_prefix)
  if out_dir and not tf.io.gfile.exists(out_dir):
    tf.io.gfile.makedirs(out_dir)

  writers = [
      tf.io.TFRecordWriter(
          '%s-%05d-of-%05d.tfrecord' % (FLAGS.output_prefix, i, FLAGS.num_shards))
      for i in range(FLAGS.num_shards)
  ]

  coco_images, coco_annotations = [], []
  id_counts = {}
  ann_id = 1
  written = 0

  for image_id, (image_path, label_path) in enumerate(pairs):
    example, anns, (w, h) = create_tf_example(
        image_path, label_path, class_names, image_id, id_counts)
    writers[image_id % FLAGS.num_shards].write(example.SerializeToString())
    written += 1
    coco_images.append({'id': image_id,
                        'file_name': os.path.basename(image_path),
                        'width': w, 'height': h})
    for a in anns:
      a['id'] = ann_id; ann_id += 1
      coco_annotations.append(a)
    if written % 500 == 0:
      logging.info('Processed %d images.', written)

  for w in writers:
    w.close()

  logging.info('Done. Wrote %d images across %d shards.', written, FLAGS.num_shards)
  # per-class object counts -- handy sanity check that all 24 ids appear
  logging.info('Per-class object counts (global id -> name -> count):')
  for cid in range(len(class_names)):
    logging.info('  %2d  %-20s %d', cid, class_names[cid], id_counts.get(cid, 0))

  if FLAGS.json_output:
    categories = [{'id': i + 1, 'name': n, 'supercategory': 'object'}
                  for i, n in enumerate(class_names)]
    coco = {'images': coco_images, 'annotations': coco_annotations,
            'categories': categories}
    with tf.io.gfile.GFile(FLAGS.json_output, 'w') as f:
      json.dump(coco, f)
    logging.info('Wrote COCO-style json to %s', FLAGS.json_output)


if __name__ == '__main__':
  app.run(main)
