import tensorflow as tf
import numpy as np
np.random.seed(2017)
import time
from model import *
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
plt.rcParams['figure.figsize'] = (12.0, 6.0) #set default size of plots
import seaborn as sns
sns.set(color_codes=True)



class Solver(object):
	"""docstring for Solver"""
	def __init__(self, data, noise, prior, gan, maxiter=4000, sample_size=10000):
		super(Solver, self).__init__()
		self.data = data
		self.noise = noise
		self.prior = prior
		self.gan = gan
		self.maxiter = maxiter
		self.sample_size = sample_size
		self.config = tf.ConfigProto()
		self.config.gpu_options.allow_growth = True

	def train(self):
		theta = tf.placeholder(tf.float32, [None, 1], 'theta')
		noise = tf.placeholder(tf.float32, [None, 3], 'noise')

		logits_real = self.gan.discriminator(theta)

		x = tf.placeholder(tf.float32, [None, 1], 'x')
		prior = self.prior.prob(x)
		logits_x = self.gan.discriminator(x, reuse=True)
		D_x = tf.nn.sigmoid(logits_x)
		

		logits_real_mean = tf.reduce_mean(logits_real)

		theta_fake = self.gan.generator(noise)
		logits_fake = self.gan.discriminator(theta_fake, reuse=True)
		logits_fake_mean = tf.reduce_mean(logits_fake)

		loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(logits_real))
		loss_fake_1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(logits_fake))
		loss_fake_2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(logits_fake))

		D_loss = tf.reduce_mean( loss_real + loss_fake_1)
		G_loss = tf.reduce_mean(loss_fake_2)

		D_opt = tf.train.AdamOptimizer(1e-4, beta1=0.5, name='D_opt')
		G_opt = tf.train.AdamOptimizer(1e-4, beta1=0.5, name='G_opt')

		D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
		G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
		D_train_step = D_opt.minimize(D_loss, var_list=D_vars)
		G_train_step = G_opt.minimize(G_loss, var_list=G_vars)

		a = np.linspace(1e-6, 30, 10000).reshape(10000,1)
		anim_frames = []
		frame_num = []
		with tf.Session(config=self.config) as sess:
			# saver = tf.train.Saver(max_to_keep=10)
			sess.run(tf.global_variables_initializer())
			pdf = sess.run(prior, {x: a})
			for step in range(self.maxiter + 1):
				data_samples = self.data.sample(100)
				noise_samples = self.noise.sample(100)
				for j in range(1):
				 	dl, lr, _ = sess.run([D_loss, logits_real_mean, D_train_step], {theta: data_samples, noise: noise_samples})

				noise_samples = self.noise.sample(100)   			
				for k in range(1):
					gl, lf, _ = sess.run([G_loss, logits_fake_mean, G_train_step], {noise: noise_samples})

				if step % 10 == 0:
					print ('Iter: [%d], loss D: [%.4f], loss G: [%4f], logits real: [%.4f], logits fake: [%.4f]' % (step, dl, gl, lr, lf))

				# if step % 1000 == 0 and step != 0:
				# 	saver.save(sess, './model/gan', global_step = step)
				# 	print ('gan-%d saved!' % (step))
					noise_samples = self.noise.sample(self.sample_size)
					gen_samples = sess.run(theta_fake, {noise: noise_samples})
					db = sess.run(D_x, {x: a})
					anim_frames.append((db, gen_samples))
					frame_num.append(step)

		self._animation(a, pdf, anim_frames, frame_num)



	


	def eval(self, version):
		noise = tf.placeholder(tf.float32, [None, 3], 'noise')
		x = tf.placeholder(tf.float32, [None, 1], 'x')
		prior = self.prior.prob(x)
		theta_gen = self.gan.generator(noise)
		with tf.Session(config=self.config) as sess:
			a = np.linspace(1e-6, 30, 10000).reshape(10000,1)
			print ('loading pre-trained model version %s' % (version))
			saver = tf.train.Saver()
			model = './model/gan-' + str(version)
			saver.restore(sess, model)
			gen_samples = sess.run(theta_gen, {noise: self.noise.sample(self.sample_size)})
			pdf = sess.run(prior, {x: a})
		self._plot(a, pdf, gen_samples, version)
		print('Done!')


	def _plot(self, a, prior, samples, version):

		plt.plot(a, prior, 'g', label='Gamma prior')
		hist, edge = np.histogram(samples, bins=100, density=True)
		b = np.linspace(edge[0], edge[-1], len(hist))
		plt.plot(b, hist, 'r', label='generated samples')
		plt.legend(ncol=2, loc=9)
		fig_name = './sample/gan-' + version
		plt.savefig(fig_name)
		plt.close()

	def _animation(self, a, prior, anim_frames, frame_num):
		f, ax = plt.subplots(figsize=(16, 8))
		f.suptitle('Generative Adversarial Network', fontsize=15)
		plt.xlabel('Data values')
		plt.ylabel('Probability density')
		ax.set_xlim(0, 30)
		ax.set_ylim(0, 1)
		line_gamma, = ax.plot([], [], 'g', label='Gamma prior')
		line_gd, = ax.plot([], [], 'r', label='generated samples')
		line_db, = ax.plot([], [], 'b', label='decision boundary')
		frame_number = ax.text(
		    0.02,
		    0.95,
		    '',
		    horizontalalignment='left',
		    verticalalignment='top',
		    transform=ax.transAxes
		)
		ax.legend(ncol=4, loc=9)

		def init():
			line_gamma.set_data([], [])
			line_gd.set_data([], [])
			line_db.set_data([], [])
			frame_number.set_text('')
			return (line_gamma, line_gd, line_db, frame_number)


		def animate(i):
			frame_number.set_text(
			    'Iter: {}/{}'.format(frame_num[i], frame_num[-1])
			)
			db, samples = anim_frames[i]
			hist, edge = np.histogram(samples, bins=100, density=True)
			b = np.linspace(edge[0], edge[-1], len(hist))
			line_gamma.set_data(a, prior)
			line_db.set_data(a, db)
			line_gd.set_data(b, hist)
			return (line_gamma, line_gd, line_db, frame_number)

		anim = animation.FuncAnimation(
		    f,
		    animate,
		    init_func=init,
		    frames=len(anim_frames),
		    blit=True
		)
		anim.save('animation_2_2.mp4', fps=10, extra_args=['-vcodec', 'libx264'])








