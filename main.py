import argparse
import tensorflow as tf

import os
from solver import Solver


parser = argparse.ArgumentParser(description='A simple demo using gan to generate gamma distributions', epilog='#' * 75)
########## Training Configuration ##########
parser.add_argument('--gpu', default='', type=str, help='gpu to use: 0, 1, 2, 3, 4.  Default: None')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate. Default: 1e-4')
parser.add_argument('--iterations', default=2000, type=int, help='num of iterations. Default: 2000')
parser.add_argument('--alpha', default=2., type=float, help='Gamma alpha. Default: 2.')
parser.add_argument('--beta', default=2., type=float, help='Gamma beta. Default: 2.')
parser.add_argument('--sample_size', default=100, type=int, help='sample size. Default: 100')
########## Directories Configuration ##########
parser.add_argument('--model_save_dir', type=str, default='checkpoints', help='directory to save model. Default: checkpoints')
parser.add_argument('--res_save_dir', type=str, default='results', help='directory to save results. Default: results')
parser.add_argument('--log_save_dir', type=str, default='logs', help='directory to save logs. Default: logs')
args = parser.parse_args()



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


leaky_relu = tf.nn.leaky_relu
class Gan(object):
    def __init__(self):
        super(Gan, self).__init__()

    def discriminator(self, x):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            h1 = tf.layers.dense(inputs=x,  units=256, activation=leaky_relu)
            h2 = tf.layers.dense(inputs=h1, units=256, activation=leaky_relu)
            logits = tf.layers.dense(inputs=h2, units=1)
            return logits


    def generator(self, z):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            h1 = tf.layers.dense(inputs=z, units=100, activation=leaky_relu)
            h2 = tf.layers.dense(inputs=h1, units=100, activation=leaky_relu)
            h3 = tf.layers.dense(inputs=h2, units=100, activation=leaky_relu)
            x = tf.layers.dense(inputs=h3, units=1, activation=tf.nn.relu)
            x = x + 1e-6 # add jitter
            return x

def main():
    
    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)
    if not os.path.exists(args.res_save_dir):
        os.mkdir(args.res_save_dir)
    if not os.path.exists(args.log_save_dir):
        os.mkdir(args.log_save_dir)

    gan = Gan()
    solver = Solver(gan, args)
    solver.train()


if __name__ == '__main__':
    main()

