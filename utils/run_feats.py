"""Extract image features with TF-Slim."""

import json
import os
import platform
import sys
import tarfile

import numpy as np
import tensorflow as tf
import tqdm
from tensorflow.contrib.slim.nets import resnet_v1, vgg
# from tensorflow.contrib.slim.python.slim.data import vgg_preprocessing
import vgg_preprocessing

from six.moves import urllib
from pycocotools.coco import COCO

# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

slim = tf.contrib.slim

__models__ = {
    'vgg_16': 'http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz',
    'vgg_19': 'http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz',
    'resnet_v1_50': 'http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz',
    'resnet_v1_101': 'http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz',
    'resnet_v1_152': 'http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz',
    'resnet_v2_50': 'http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz',
    'resnet_v2_101': 'http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz',
    'resnet_v2_152': 'http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz',
}

if platform.system() == 'Windows':
    __checkdir__ = 'G:\\image_caption\\pretrain'
else:
    __checkdir__ = '/mnt/ht/image_caption/pretrain'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ann_json', '', """Input json file.""")
tf.app.flags.DEFINE_string('model', 'vgg_16', """Model name.""")
tf.app.flags.DEFINE_string('output_path', '', """Path prefix for output.""")
tf.app.flags.DEFINE_string('image_folder', '', """folder of images.""")


def get_sess_config():
    """Session configs."""
    conf = tf.ConfigProto(
        log_device_placement=True,
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.4,
            allow_growth=True
        ),
        device_count={'CPU': 1}
    )
    return conf


def download_and_uncompress_tarball(tarball_url, dataset_dir):
  """Downloads the `tarball_url` and uncompresses it locally.

  Args:
    tarball_url: The URL of a tarball file.
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = tarball_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()
  filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
  print()
  statinfo = os.stat(filepath)
  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(fileepath, 'r:gz').extractall(dataset_dir)


def get_pretrained_model(model):
    assert model in __models__, "Model %s is not supported." % model
    model_dir = os.path.join(__checkdir__, model)
    if not tf.gfile.Exists(model_dir):
        tf.gfile.MakeDirs(model_dir)
    download_and_uncompress_tarball(__models__[model], model_dir)


def preprocess_image(image, height, width):
    processed_image = vgg_preprocessing.preprocess_image(image, height, width, is_training=False)
    processed_image = tf.expand_dims(processed_image, 0)
    return processed_image


def get_init_fn(model):
    assert model in __models__, "Model [%s] not found."
    model_family = model.split('-')[0]    # Should be 'vgg' or 'resnet'
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(__checkdir__, model, model+'.ckpt'),
        slim.get_model_variables(model_family)
    )
    return init_fn


def get_images(filenames): #image_path_pattern):
    #filenames = tf.train.match_filenames_once(image_path_pattern)
    count_num_files = tf.size(filenames)
    filename_queue = tf.train.string_input_producer(filenames)

    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value, channels=3)

    return image, count_num_files


def get_layer_output(model, endpoint):
    if model == 'vgg_16':
        fc_out = tf.reshape(endpoint['vgg_16/fc7'], [-1])
        att_out = endpoint['vgg_16/pool5']
        return fc_out, att_out


def main(argv):
    """Main entrance."""
    model = 'vgg_16'

    coco = COCO(FLAGS.ann_json)
    img_ids = coco.getImgIds()

    image_id_to_line_no = {i:n for n, i in enumerate(coco.getImgIds())}
    path_to_line_no = {
        os.path.join(FLAGS.image_folder, coco.imgs[i]['file_name']): image_id_to_line_no[i] for i in image_id_to_line_no.keys()}

    num_files = len(path_to_line_no)

    with open('../save/imgid2lineno.json', 'w') as f:
        json.dump(image_id_to_line_no, f)

    fc_path = FLAGS.output_path + '_fc.mmp'
    att_path = FLAGS.output_path + '_att.mmp'

    fc_shape = (num_files,) + (4096,)
    att_shape = (num_files,) + (7, 7, 512)

    mmp_fc = np.memmap(fc_path, dtype='float32', mode='w+', shape=fc_shape)
    mmp_att = np.memmap(att_path, dtype='float32', mode='w+', shape=att_shape)
    img_size = vgg.vgg_16.default_image_size


    with tf.Graph().as_default():
        inp = tf.placeholder(dtype=tf.string, shape=[])

        image = tf.image.decode_jpeg(inp, channels=3)
        processed_image = preprocess_image(image, img_size, img_size)

        import ipdb; ipdb.set_trace()

        # slim
        with slim.arg_scope(vgg.vgg_arg_scope()):
            _, endpoint = vgg.vgg_16(processed_image, num_classes=1000, is_training=False)

        fc_feat, att_feat = get_layer_output(model, endpoint)

        init_fn = get_init_fn(model)
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session(config=get_sess_config()) as sess:
            sess.run(init)
            init_fn(sess)

            for img in tqdm.tqdm(path_to_line_no.keys(), total=num_files):
                img_data = tf.gfile.FastGFile(img, 'rb').read()
                fc, att = sess.run([fc_feat, att_feat], feed_dict={inp: img_data})
                idx = path_to_line_no[img]
                mmp_fc[idx] = fc
                mmp_att[idx] = att

            del mmp_fc, mmp_att


if __name__ == '__main__':
    tf.app.run()
