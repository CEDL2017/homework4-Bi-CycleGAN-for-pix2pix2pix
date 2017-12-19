import tensorflow as tf
from model import CycleGAN, BiCycleGAN
from reader import Reader
from datetime import datetime
import os
import logging
from utils import ImagePool

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 256, 'image size, default: 256')
tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_integer('lambda1', 10.0,
                        'weight for forward cycle loss (X->Y->X), default: 10.0')
tf.flags.DEFINE_integer('lambda2', 10.0,
                        'weight for backward cycle loss (Y->X->Y), default: 10.0')
tf.flags.DEFINE_float('lambda_identity', 0.5,
                        'weight for identity mapping loss, default: 0.5')
tf.flags.DEFINE_float('learning_rate', 2e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')

tf.flags.DEFINE_string('X', 'data/tfrecords/apple.tfrecords',
                       'X tfrecords file for training, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('Y', 'data/tfrecords/orange.tfrecords',
                       'Y tfrecords file for training, default: data/tfrecords/orange.tfrecords')
tf.flags.DEFINE_string('Z', '',
                       'Z tfrecords file for training')
tf.flags.DEFINE_string('load_model', None,
                        'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)

MAX_STEP = 200000

def train():
  if FLAGS.load_model is not None:
    checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
  else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = "checkpoints/{}".format(current_time)
    try:
      os.makedirs(checkpoints_dir)
    except os.error:
      pass

  graph = tf.Graph()
  with graph.as_default():
    bi_cycle_gan = BiCycleGAN(
        X_train_file=FLAGS.X,
        Y_train_file=FLAGS.Y,
        Z_train_file=FLAGS.Z,
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.image_size,
        use_lsgan=FLAGS.use_lsgan,
        norm=FLAGS.norm,
        lambda1=FLAGS.lambda1,
        lambda2=FLAGS.lambda2,
        lambda_identity=FLAGS.lambda_identity,
        learning_rate=FLAGS.learning_rate,
        beta1=FLAGS.beta1,
        ngf=FLAGS.ngf
    )
    XY_G_loss, XY_D_Y_loss, XY_F_loss, XY_D_X_loss, XY_fake_y, XY_fake_x, \
        XZ_G_loss, XZ_D_Z_loss, XZ_F_loss, XZ_D_X_loss, XZ_fake_z, XZ_fake_x = bi_cycle_gan.model()
    optimizers = bi_cycle_gan.optimize(XY_G_loss, XY_D_Y_loss, XY_F_loss, XY_D_X_loss, XZ_G_loss, XZ_D_Z_loss, XZ_F_loss, XZ_D_X_loss)

    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
    saver = tf.train.Saver()

  with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    if FLAGS.load_model is not None:
      checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
      meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
      restore = tf.train.import_meta_graph(meta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
      step = int(meta_graph_path.split("-")[2].split(".")[0])
    else:
      sess.run(tf.global_variables_initializer())
      step = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      XY_fake_Y_pool = ImagePool(FLAGS.pool_size)
      XY_fake_X_pool = ImagePool(FLAGS.pool_size)
      XZ_fake_Z_pool = ImagePool(FLAGS.pool_size)
      XZ_fake_X_pool = ImagePool(FLAGS.pool_size)

      while not coord.should_stop() and step < MAX_STEP:
        # get previously generated images
        XY_fake_y_val, XY_fake_x_val, XZ_fake_z_val, XZ_fake_x_val = sess.run([XY_fake_y, XY_fake_x, XZ_fake_z, XZ_fake_x])

        # train
        _, XY_G_loss_val, XY_D_Y_loss_val, XY_F_loss_val, XY_D_X_loss_val, XZ_G_loss_val, XZ_D_Z_loss_val, XZ_F_loss_val, XZ_D_X_loss_val, summary = (
              sess.run(
                  [optimizers, XY_G_loss, XY_D_Y_loss, XY_F_loss, XY_D_X_loss, XZ_G_loss, XZ_D_Z_loss, XZ_F_loss, XZ_D_X_loss, summary_op],
                  feed_dict={bi_cycle_gan.XY_fake_y: XY_fake_Y_pool.query(XY_fake_y_val),
                             bi_cycle_gan.XY_fake_x: XY_fake_X_pool.query(XY_fake_x_val),
                             bi_cycle_gan.XZ_fake_z: XZ_fake_Z_pool.query(XZ_fake_z_val),
                             bi_cycle_gan.XZ_fake_x: XZ_fake_X_pool.query(XZ_fake_x_val)}
              )
        )

        train_writer.add_summary(summary, step)
        train_writer.flush()

        if step % 100 == 0:
          logging.info('-----------Step %d:-------------' % step)
          logging.info('  XY_G_loss   : {}'.format(XY_G_loss_val))
          logging.info('  XY_D_Y_loss : {}'.format(XY_D_Y_loss_val))
          logging.info('  XY_F_loss   : {}'.format(XY_F_loss_val))
          logging.info('  XY_D_X_loss : {}'.format(XY_D_X_loss_val))
          logging.info('  XZ_G_loss   : {}'.format(XZ_G_loss_val))
          logging.info('  XZ_D_Z_loss : {}'.format(XZ_D_Z_loss_val))
          logging.info('  XZ_F_loss   : {}'.format(XZ_F_loss_val))
          logging.info('  XZ_D_X_loss : {}'.format(XZ_D_X_loss_val))

        if step % 10000 == 0:
          save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
          logging.info("Model saved in file: %s" % save_path)

        step += 1

    except KeyboardInterrupt:
      logging.info('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
      logging.info("Model saved in file: %s" % save_path)
      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

def main(unused_argv):
  train()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()
