r"""YOLO (nested-per-class) -> EfficientDet TFRecord, v2.

v2 changes over v1:
  - Re-encode every image as JPEG via PIL before writing. This normalises
    WebP / TIFF / AVIF / weird-extension files into something TF's
    DecodeImage op can read (JPEG/PNG/GIF/BMP).
  - Catch decode errors per image and skip + log instead of aborting.
  - Track and report the original format distribution + skip count at the end.

Usage is identical to v1:
    python yolo_nested_to_tfrecord_v2.py \
        --split_dir /path/to/train \
        --classes_file classes.txt \
        --output_prefix tfrecord/train \
        --num_shards 32 \
        --json_output tfrecord/train.json
"""

import collections
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
                    'Root dir of one split (contains per-class subfolders).')
flags.DEFINE_string('classes_file', None,
                    'One class name per line; line index = 0-based class id.')
flags.DEFINE_string('output_prefix', None, 'e.g. tfrecord/train.')
flags.DEFINE_integer('num_shards', 32, 'Number of output TFRecord shards.')
flags.DEFINE_string('json_output', None, 'Optional COCO-style json output.')
flags.DEFINE_integer('jpeg_quality', 95, 'JPEG quality for re-encoded images.')
FLAGS = flags.FLAGS

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff',
            '.gif', '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP')


def _int64_feature(v):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))


def _int64_list_feature(v):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=v))


def _bytes_feature(v):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def _bytes_list_feature(v):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=v))


def _float_list_feature(v):
  return tf.train.Feature(float_list=tf.train.FloatList(value=v))


def load_class_names(path):
  with tf.io.gfile.GFile(path, 'r') as f:
    return [line.strip() for line in f if line.strip()]


def collect_pairs(split_dir):
  pairs = []
  class_folders = sorted(
      d.rstrip('/') for d in tf.io.gfile.listdir(split_dir)
      if tf.io.gfile.isdir(os.path.join(split_dir, d.rstrip('/'))))
  for cf in class_folders:
    img_dir = os.path.join(split_dir, cf, 'images')
    lbl_dir = os.path.join(split_dir, cf, 'labels')
    if not tf.io.gfile.exists(img_dir):
      logging.warning('No images/ in %s, skipping.', cf)
      continue
    label_by_stem = {}
    if tf.io.gfile.exists(lbl_dir):
      for lf in tf.io.gfile.listdir(lbl_dir):
        if lf.endswith('.txt'):
          label_by_stem[os.path.splitext(lf)[0]] = os.path.join(lbl_dir, lf)
    for imf in tf.io.gfile.listdir(img_dir):
      if not imf.endswith(IMG_EXTS):
        continue
      pairs.append((os.path.join(img_dir, imf),
                    label_by_stem.get(os.path.splitext(imf)[0])))
  return pairs


def yolo_line_to_corners(parts):
  cls = int(float(parts[0]))
  xc, yc, w, h = (float(p) for p in parts[1:5])
  return (cls,
          max(0.0, xc - w / 2.0), max(0.0, yc - h / 2.0),
          min(1.0, xc + w / 2.0), min(1.0, yc + h / 2.0))


def load_and_reencode_as_jpeg(image_path, fmt_counter, quality):
  """Open with PIL (which understands many formats), return JPEG bytes + size.

  Returns (jpeg_bytes, width, height, original_format).
  Raises on unreadable / unsupported files.
  """
  with tf.io.gfile.GFile(image_path, 'rb') as f:
    raw = f.read()
  img = PIL.Image.open(io.BytesIO(raw))
  original_format = img.format or 'UNKNOWN'
  fmt_counter[original_format] += 1
  # convert palette / RGBA / L to RGB so JPEG encoder is happy
  if img.mode not in ('RGB',):
    img = img.convert('RGB')
  buf = io.BytesIO()
  img.save(buf, format='JPEG', quality=quality)
  return buf.getvalue(), img.size[0], img.size[1], original_format


def create_tf_example(jpeg_bytes, width, height, image_path, label_path,
                      class_names, image_id, id_counts):
  key = hashlib.sha256(jpeg_bytes).hexdigest()
  xmins, ymins, xmaxs, ymaxs = [], [], [], []
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
          logging.warning('cls id %d out of range in %s; skipping box.',
                          cls, label_path)
          continue
        id_counts[cls] = id_counts.get(cls, 0) + 1
        xmins.append(xmin); ymins.append(ymin)
        xmaxs.append(xmax); ymaxs.append(ymax)
        class_ids.append(cls + 1)
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

  feat = {
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/filename': _bytes_feature(os.path.basename(image_path).encode('utf8')),
      'image/source_id': _bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256': _bytes_feature(key.encode('utf8')),
      'image/encoded': _bytes_feature(jpeg_bytes),
      'image/format': _bytes_feature(b'jpeg'),
      'image/object/bbox/xmin': _float_list_feature(xmins),
      'image/object/bbox/xmax': _float_list_feature(xmaxs),
      'image/object/bbox/ymin': _float_list_feature(ymins),
      'image/object/bbox/ymax': _float_list_feature(ymaxs),
      'image/object/class/text': _bytes_list_feature(class_text),
      'image/object/class/label': _int64_list_feature(class_ids),
      'image/object/area': _float_list_feature(areas),
      'image/object/is_crowd': _int64_list_feature(is_crowd),
  }
  return (tf.train.Example(features=tf.train.Features(feature=feat)),
          coco_annotations)


def main(_):
  for flag in ('split_dir', 'classes_file', 'output_prefix'):
    if getattr(FLAGS, flag) is None:
      raise ValueError('--%s is required' % flag)
  class_names = load_class_names(FLAGS.classes_file)
  logging.info('Loaded %d class names.', len(class_names))

  pairs = collect_pairs(FLAGS.split_dir)
  logging.info('Collected %d image/label pairs.', len(pairs))
  if not pairs:
    raise ValueError('No images found under --split_dir.')

  out_dir = os.path.dirname(FLAGS.output_prefix)
  if out_dir and not tf.io.gfile.exists(out_dir):
    tf.io.gfile.makedirs(out_dir)
  writers = [tf.io.TFRecordWriter('%s-%05d-of-%05d.tfrecord' %
                                   (FLAGS.output_prefix, i, FLAGS.num_shards))
             for i in range(FLAGS.num_shards)]

  coco_images, coco_annotations = [], []
  id_counts = {}
  fmt_counter = collections.Counter()
  bad_files = []
  ann_id = 1
  written = 0

  for image_id, (image_path, label_path) in enumerate(pairs):
    try:
      jpeg_bytes, w, h, _ = load_and_reencode_as_jpeg(
          image_path, fmt_counter, FLAGS.jpeg_quality)
    except Exception as e:                            # broad on purpose
      logging.warning('skip unreadable image %s: %s', image_path, e)
      bad_files.append((image_path, str(e)))
      continue
    ex, anns = create_tf_example(
        jpeg_bytes, w, h, image_path, label_path,
        class_names, image_id, id_counts)
    writers[image_id % FLAGS.num_shards].write(ex.SerializeToString())
    written += 1
    coco_images.append({'id': image_id,
                        'file_name': os.path.basename(image_path),
                        'width': w, 'height': h})
    for a in anns:
      a['id'] = ann_id; ann_id += 1
      coco_annotations.append(a)
    if written % 500 == 0:
      logging.info('processed %d images', written)

  for w in writers: w.close()

  logging.info('Done. Wrote %d images; skipped %d bad files; %d shards.',
               written, len(bad_files), FLAGS.num_shards)
  logging.info('Original format distribution: %s', dict(fmt_counter))
  if bad_files:
    bad_log = FLAGS.output_prefix + '.bad_files.txt'
    with open(bad_log, 'w') as f:
      for p, err in bad_files:
        f.write('%s\t%s\n' % (p, err))
    logging.info('List of skipped files written to %s', bad_log)
  logging.info('Per-class counts:')
  for cid in range(len(class_names)):
    logging.info('  %2d  %-25s %d', cid, class_names[cid], id_counts.get(cid, 0))

  if FLAGS.json_output:
    categories = [{'id': i + 1, 'name': n, 'supercategory': 'object'}
                  for i, n in enumerate(class_names)]
    with tf.io.gfile.GFile(FLAGS.json_output, 'w') as f:
      json.dump({'images': coco_images, 'annotations': coco_annotations,
                 'categories': categories}, f)
    logging.info('Wrote COCO json to %s', FLAGS.json_output)


if __name__ == '__main__':
  app.run(main)
