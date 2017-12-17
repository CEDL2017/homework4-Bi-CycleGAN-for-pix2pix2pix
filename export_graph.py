""" Freeze variables and convert 2 generator networks to 2 GraphDef files.
This makes file size smaller and can be used for inference in production.
An example of command-line usage is:
python export_graph.py --checkpoint_dir checkpoints/20170424-1152 \
                       --XtoY_model apple2orange.pb \
                       --YtoX_model orange2apple.pb \
                       --image_size 256
"""

import tensorflow as tf
import os
from tensorflow.python.tools.freeze_graph import freeze_graph
from model import CycleGAN, BiCycleGAN
import utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint_dir', '', 'checkpoints directory path')
tf.flags.DEFINE_string('XtoY_model', 'apple2orange.pb', 'XtoY model name, default: apple2orange.pb')
tf.flags.DEFINE_string('YtoX_model', 'orange2apple.pb', 'YtoX model name, default: orange2apple.pb')
tf.flags.DEFINE_string('XtoZ_model', '', 'XtoZ model name')
tf.flags.DEFINE_string('ZtoX_model', '', 'ZtoX model name')
tf.flags.DEFINE_integer('image_size', '256', 'image size, default: 256')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

def export_graph(model_name, XtoY=2, XtoZ=2):
  graph = tf.Graph()

  with graph.as_default():
    bi_cycle_gan = BiCycleGAN(ngf=FLAGS.ngf, norm=FLAGS.norm, image_size=FLAGS.image_size)

    input_image = tf.placeholder(tf.float32, shape=[FLAGS.image_size, FLAGS.image_size, 3], name='input_image')
    bi_cycle_gan.model()
    if XtoY == 2:
      output_image = bi_cycle_gan.XY_G.sample(tf.expand_dims(input_image, 0))
    elif XtoY == 1:
      output_image = bi_cycle_gan.XY_F.sample(tf.expand_dims(input_image, 0))
    elif XtoZ == 2:
      output_image = bi_cycle_gan.XZ_G.sample(tf.expand_dims(input_image, 0))
    elif XtoZ == 1:
      output_image = bi_cycle_gan.XZ_F.sample(tf.expand_dims(input_image, 0))

    output_image = tf.identity(output_image, name='output_image')
    restore_saver = tf.train.Saver()
    export_saver = tf.train.Saver()

  with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    restore_saver.restore(sess, latest_ckpt)
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [output_image.op.name])

    tf.train.write_graph(output_graph_def, 'pretrained', model_name, as_text=False)

def main(unused_argv):
  print('Export XtoY model...', FLAGS.XtoY_model)
  export_graph(FLAGS.XtoY_model, XtoY=2, XtoZ=0)
  print('Export YtoX model...', FLAGS.YtoX_model)
  export_graph(FLAGS.YtoX_model, XtoY=1, XtoZ=0)

  print('Export XtoZ model...', FLAGS.XtoZ_model)
  export_graph(FLAGS.XtoZ_model, XtoY=0, XtoZ=2)
  print('Export ZtoX model...', FLAGS.ZtoX_model)
  export_graph(FLAGS.ZtoX_model, XtoY=0, XtoZ=1)

if __name__ == '__main__':
  tf.app.run()
