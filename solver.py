import tensorflow as tf
import numpy as np
np.random.seed(2017)
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
plt.rcParams['figure.figsize'] = (12.0, 6.0) # set default size of plots
import seaborn as sns
sns.set(color_codes=True)
import os
from tensorflow.contrib.distributions import Gamma



def write(path, file, log, mode='a'):
    print (log)
    with open(os.path.join(path, file), mode) as f:
        f.write(log + '\n')

class Solver(object):
    """docstring for Solver"""
    def __init__(self, gan, args):
        super(Solver, self).__init__()
        self.alpha, self.beta = int(args.alpha), int(args.beta)
        self.gamma = Gamma(concentration=args.alpha, rate=args.beta)
        self.gan = gan
        self.iterations = args.iterations
        self.lr = args.lr
        self.model_save_dir = args.model_save_dir
        self.res_save_dir = args.res_save_dir
        self.log_save_dir = args.log_save_dir
        self.sample_size = args.sample_size
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.eps_dim = 3
        

    def train(self):
        
        x = np.linspace(1e-6, 20, 10000, dtype=np.float32)[:, None]
        theta = self.gamma.sample(self.sample_size)[:, None]
        probs = self.gamma.prob(x)

        logits_real = self.gan.discriminator(theta)
        logits_real_mean = tf.reduce_mean(logits_real)

        noise = tf.random_normal((self.sample_size, self.eps_dim))
        theta_fake = self.gan.generator(noise)

        noise2 = tf.random_normal((10000, self.eps_dim))
        theta_gen_ = self.gan.generator(noise2)



        logits_fake = self.gan.discriminator(theta_fake)
        logits_fake_mean = tf.reduce_mean(logits_fake)

        loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(logits_real))
        loss_fake_1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(logits_fake))
        loss_fake_2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(logits_fake))

        D_loss = tf.reduce_mean( loss_real + loss_fake_1)
        G_loss = tf.reduce_mean(loss_fake_2)

        D_opt = tf.train.AdamOptimizer(self.lr, beta1=0.5, name='D_opt')
        G_opt = tf.train.AdamOptimizer(self.lr, beta1=0.5, name='G_opt')

        D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        D_train_step = D_opt.minimize(D_loss, var_list=D_vars)
        G_train_step = G_opt.minimize(G_loss, var_list=G_vars)

        with tf.Session(config=self.config) as sess:
            saver = tf.train.Saver(max_to_keep=10)
            sess.run(tf.global_variables_initializer())
            pdf = sess.run(probs)
            anim_frames = []
            frame_num = []
            for step in range(self.iterations + 1):
                for j in range(1):
                    dl, lr, _ = sess.run([D_loss, logits_real_mean, D_train_step])            
                for k in range(1):
                    gl, lf, _ = sess.run([G_loss, logits_fake_mean, G_train_step])

                if step % 50 == 0:
                    theta_gen = sess.run(theta_gen_)
                    anim_frames.append(theta_gen)
                    frame_num.append(step)

                if step % 100 == 0:
                    log = 'Iter: [{}], loss D: [{:3f}], loss G: [{:3f}], logits real: [{:3f}], logits fake: [{:3f}]'.format(step, dl, gl, lr, lf)
                    write(self.log_save_dir, 'log_gamma_alpha_{}_beta_{}.out'.format(self.alpha, self.beta), log)

                if step % 1000 == 0 and step != 0:
                    saver.save(sess, '{}/gan_alpha_{}_beta_{}'.format(self.model_save_dir, self.alpha, self.beta), global_step = step)
                    print ('gan-{}_alpha_{}_beta_{}.ckpt saved......'.format(step, self.alpha, self.beta))
            print ('creating the final video......')
            self._animation(x, pdf, anim_frames, frame_num)
            print ('Done......!')


    def _animation(self, a, pdf, anim_frames, frame_num):
        f, ax = plt.subplots(figsize=(16, 8))
        f.suptitle('Gamma', fontsize=15)
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 1)
        line_posterior, = ax.plot([], [], 'g', label='gamma distribution')
        line_gd, = ax.plot([], [], 'r', label='generated samples')
    #     hist_gd, = ax.hist([], 100, normed=1, alpha=0.8, color='r', label='generated samples')
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
            line_posterior.set_data([], [])
            line_gd.set_data([], [])
            frame_number.set_text('')
            return (line_posterior, line_gd, frame_number)


        def animate(i):
            frame_number.set_text(
                'Iter: {}/{}'.format(frame_num[i], frame_num[-1])
            )
            samples = anim_frames[i]
            hist, edge = np.histogram(samples, bins=100, density=True)
            b = np.linspace(edge[0], edge[-1], len(hist))
            line_posterior.set_data(a, pdf)
            line_gd.set_data(b, hist)
            return (line_posterior, line_gd, frame_number)

        anim = animation.FuncAnimation(
            f,
            animate,
            init_func=init,
            frames=len(anim_frames),
            blit=True
        )
        anim.save('{}/gamma_alpha_{}_beta_{}.mp4'.format(self.res_save_dir, self.alpha, self.beta), fps=10, extra_args=['-vcodec', 'libx264'])








