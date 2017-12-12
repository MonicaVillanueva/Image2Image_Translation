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

	def __init__(self, imgDim=(128,128), numClass=5):
		# Weights initialised
		self.imgDim = imgDim;

		self.downSampling1_W = weight_variable([7, 7, 3 + numClass, 64])
		self.downSampling1_b = bias_variable([64])

		self.downSampling2_W = weight_variable([4, 4, 64, 128])
		self.downSampling2_b = bias_variable([128])

		self.downSampling3_W = weight_variable([4, 4, 128, 256])
		self.downSampling3_b = bias_variable([256])

		self.residualBlock1 = ResidualBlock();
		self.residualBlock2 = ResidualBlock();
		self.residualBlock3 = ResidualBlock();
		self.residualBlock4 = ResidualBlock();
		self.residualBlock5 = ResidualBlock();
		self.residualBlock6 = ResidualBlock();

		self.upSampling1_W = weight_variable([4, 4, 256, 128])
		self.upSampling1_b = bias_variable([128])

		self.upSampling2_W = weight_variable([4, 4, 128, 64])
		self.upSampling2_b = bias_variable([64])

		self.upSampling3_W = weight_variable([7, 7, 64, 3])
		self.upSampling3_b = bias_variable([3])


	def forward(self, x, c):
		# x has to contain already the labels concatenated

		X = tf.placeholder(tf.float32, shape=[None, self.imgDim[0], self.imgDim[1], len(c)])

		Y_downSampling1 = tf.nn.relu(tf.nn.conv2d(X, self.downSampling1_W, strides=[1, 1, 1, 1], padding='SAME') + self.downSampling1_b)
		Y_downSampling1_norm = tf.contrib.layers.batch_norm(Y_downSampling1)

		Y_downSampling2 = tf.nn.relu(tf.nn.conv2d(Y_downSampling1_norm, self.downSampling2_W, strides=[1, 2, 2, 1], padding='SAME') + self.downSampling2_b)
		Y_downSampling2_norm = tf.contrib.layers.batch_norm(Y_downSampling2)

		Y_downSampling3 = tf.nn.relu(tf.nn.conv2d(Y_downSampling2_norm, self.downSampling3_W, strides=[1, 2, 2, 1], padding='SAME') + self.downSampling3_b)
		Y_downSampling3_norm = tf.contrib.layers.batch_norm(Y_downSampling3)

		Y_residual1 = self.residualBlock1.forward(Y_downSampling3_norm)
		Y_residual2 = self.residualBlock2.forward(Y_residual1)
		Y_residual3 = self.residualBlock3.forward(Y_residual2)
		Y_residual4 = self.residualBlock4.forward(Y_residual3)
		Y_residual5 = self.residualBlock5.forward(Y_residual4)
		Y_residual6 = self.residualBlock6.forward(Y_residual5)

		Y_upSampling1 = tf.nn.relu(tf.nn.conv2d_transpose(Y_residual6, self.upSampling1_W, strides=[1, 2, 2, 1], padding='SAME') + self.upSampling1_b)
		Y_upSampling1_norm = tf.contrib.layers.batch_norm(Y_upSampling1)

		Y_upSampling2 = tf.nn.relu(tf.nn.conv2d_transpose(Y_upSampling1_norm, self.upSampling2_W, strides=[1, 2, 2, 1], padding='SAME') + self.upSampling2_b)
		Y_upSampling2_norm = tf.contrib.layers.batch_norm(Y_upSampling2)

		Y_upSampling3 = tf.nn.tanh(tf.nn.conv2d(Y_upSampling2_norm, self.upSampling3_W, strides=[1, 1, 1, 1], padding='SAME') + self.upSampling3_b)

	def train_step(self, x, c, lr=0.0001):
		# TODO: write this pseoudocode
		fake = self.forward(...)
		D.forward(fake)

		recLoss = .....

		# TODO: add decay
		return train_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.999).minimize(recLoss)



# class Discriminator():
    
# 	def __init__(self, imageSize=128, convDim=64, numClass=5):

# 	def forward(self, x):
