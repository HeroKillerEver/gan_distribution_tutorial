import tensorflow as tf
from tensorflow.contrib.distributions import Gamma, MultivariateNormalDiag
import numpy as np 
from utils import *

class Gan(object):
	"""docstring for Gan"""
	def __init__(self, data_dim=1):
		super(Gan, self).__init__()
		self.data_dim = data_dim
	

	def discriminator(self, x, reuse=False):
		with tf.variable_scope('discriminator', reuse=reuse):
			h1 = tf.layers.dense(inputs=x,  units=256, activation=leaky_relu)
# 			h1_norm = tf.layers.batch_normalization(inputs=h1, training=self.is_training)
# 			h2 = tf.layers.dense(inputs=h1_norm, units=20, activation=leaky_relu)
			h2 = tf.layers.dense(inputs=h1, units=256, activation=leaky_relu)
# 			h2_norm = tf.layers.batch_normalization(inputs=h2, training=self.is_training)
# 			logits = tf.layers.dense(inputs=h2_norm, units=1)
			logits = tf.layers.dense(inputs=h2, units=1)
			return logits


	def generator(self, z, reuse=False):
		with tf.variable_scope('generator', reuse=reuse):
			h1 = tf.layers.dense(inputs=z, units=100, activation=leaky_relu)
# 			h1_norm = tf.layers.batch_normalization(inputs=h1, training=self.is_training)
# 			h2 = tf.layers.dense(inputs=h1_norm, units=20, activation=leaky_relu)
			h2 = tf.layers.dense(inputs=h1, units=100, activation=leaky_relu)
# 			h2_norm = tf.layers.batch_normalization(inputs=h2, training=self.is_training)
# 			h3 = tf.layers.dense(inputs=h2_norm, units=20, activation=leaky_relu)
			h3 = tf.layers.dense(inputs=h2, units=100, activation=leaky_relu)
# 			h3_norm = tf.layers.batch_normalization(inputs=h3, training=self.is_training)
# 			x = tf.layers.dense(inputs=h3_norm, units=1,activation=tf.nn.softplus)
			x = tf.layers.dense(inputs=h3, units=self.data_dim, activation=tf.nn.relu)
			x = jitter(x)
			return x
		
class PriorGamma(object):
	"""docstring for PriorGamma"""
	def __init__(self, shape=2., scale=2.):
		super(PriorGamma, self).__init__()
		self.alpha = shape
		self.beta = 1. / scale
		self.G = Gamma(self.alpha, self.beta)
	def log_prob(self, x):
		return self.G.log_prob(x)
    
	def prob(self, x):
		return self.G.prob(x)


class PriorGauss(object):
	"""docstring for PriorGauss"""
	def __init__(self, mu=[2., 2.], diag_stdev = [.5, .5]):
		super(PriorGauss, self).__init__()
		self.mu = mu
		self.diag_stdev = diag_stdev
		self.mvn = MultivariateNormalDiag(self.mu, self.diag_stdev)

	def log_prob(self, x):
		return self.mvn.log_prob(x)

	def prob(self, x):
		return self.mvn.prob(x)

		


class DataGamma(object):
	"""docstring for DataGamma"""
	def __init__(self, shape=2, scale=2, out_dim=1):
		super(DataGamma, self).__init__()
		self.shape = shape
		self.scale = scale
		self.out_dim = out_dim
	def sample(self, N):
		samples = np.random.gamma(self.shape, self.scale, [N, self.out_dim])
		return samples

class DataNormal(object):
	"""docstring for DataNormal"""
	def __init__(self, mu=[2., 2.], diag_stdev = [.5, .5], data_dim=2):
		super(DataNormal, self).__init__()
		self.mu = np.array(mu)
		self.diag_stdev = np.array(diag_stdev)
		self.data_dim = data_dim
	def sample(self, N):
		samples = np.random.normal(size=[N, self.data_dim]) * self.diag_stdev + self.mu
		return samples
		

class Noise(object):
	"""docstring for Noise"""
	def __init__(self, mean=0, scale=1, noise_dim=3):
		super(Noise, self).__init__()
		self.mean = mean
		self.scale = scale
		self.noise_dim = noise_dim
	def sample(self, N):
		return np.random.normal(self.mean, self.scale, [N, self.noise_dim])
		








		
		





