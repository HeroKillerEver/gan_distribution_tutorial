import tensorflow as tf
import os
from model import *
from solver import Solver
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', "train or eval")
flags.DEFINE_string('version', '10000', 'version to draw figure')
flags.DEFINE_string('model_save_path', 'model', "directory for saving the model")
flags.DEFINE_string('sample_save_path', 'sample', "directory for saving the sampled images")
flags.DEFINE_float('shape', 2., 'Gamma shape')
flags.DEFINE_float('scale', 2., 'Gamma scale')
FLAGS = flags.FLAGS

def main(_):
	data = DataGamma(shape=FLAGS.shape, scale=FLAGS.scale)
	noise = Noise()
	prior = PriorGamma(shape=FLAGS.shape, scale=FLAGS.scale)
	gan = Gan()
	solver = Solver(data, noise, prior, gan)

	# create directories if not exist
	if not tf.gfile.Exists(FLAGS.model_save_path):
		tf.gfile.MakeDirs(FLAGS.model_save_path)

	if not tf.gfile.Exists(FLAGS.sample_save_path):
		tf.gfile.MakeDirs(FLAGS.sample_save_path)



	if FLAGS.mode == 'train':
		solver.train()
	elif FLAGS.mode == 'eval':
		solver.eval(FLAGS.version)
	else:
		raise ValueError('mode is only train or eval')


if __name__ == '__main__':
	tf.app.run()

