import tensorflow as tf
import numpy as np


def leaky_relu(x, alpha = 0.2):
	return tf.maximum(x, alpha * x)

def jitter(x):
    return x + 1e-6