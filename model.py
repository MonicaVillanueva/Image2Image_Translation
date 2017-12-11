import tensorflow as tf
import numpy as np

# FLAG: use Instance normalization instead of batch normalization

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

class ResidualBlock():
    """Residual Block."""
    def __init__(self, dim_in=256, dim_out=256):
		self.residualBlock1_W = weight_variable([3, 3, dim_in, dim_out])
		self.residualBlock1_b = bias_variable([dim_out])

		self.residualBlock2_W = weight_variable([3, 3, dim_out, dim_out])
		self.residualBlock2_b = bias_variable([dim_out])

    def forward(self, x):
    	Y1 = tf.nn.conv2d(x, self.residualBlock1_W, strides=[1, 1, 1, 1], padding='SAME') + self.residualBlock1_b
    	Y1_norm = tf.contrib.layers.batch_norm(Y1)

    	Y1_relu = tf.nn.relu(Y1_norm)

    	Y2 = tf.nn.conv2d(Y1_relu, self.residualBlock2_W, strides=[1, 1, 1, 1], padding='SAME') + self.residualBlock2_b
    	Y2_norm = tf.contrib.layers.batch_norm(Y2)

        return x + Y2_norm


class Generator():

	def __init__(self, convDim=64, numClass=5):
		# Weights initialised

		downSampling1_W = weight_variable([7, 7, 3 + numClass, 64])
		downSampling1_b = bias_variable([64])

		downSampling2_W = weight_variable([4, 4, 64, 128])
		downSampling2_b = bias_variable([128])

		downSampling3_W = weight_variable([4, 4, 128, 256])
		downSampling3_b = bias_variable([256])

		residualBlock1 = ResidualBlock();
		residualBlock2 = ResidualBlock();
		residualBlock3 = ResidualBlock();
		residualBlock4 = ResidualBlock();
		residualBlock5 = ResidualBlock();
		residualBlock6 = ResidualBlock();

		upSampling1_W = weight_variable([4, 4, 256, 128])
		upSampling1_b = bias_variable([128])

		upSampling2_W = weight_variable([4, 4, 128, 64])
		upSampling2_b = bias_variable([64])

		upSampling3_W = weight_variable([7, 7, 64, 3])
		upSampling3_b = bias_variable([3])


	def forward(self, x, c):
		# TODO: concatenate c to x


class Discriminator():
    
	def __init__(self, imageSize=128, convDim=64, numClass=5):

	def forward(self, x):
