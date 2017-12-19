import tensorflow as tf
import numpy as np

# FLAG: use Instance normalization instead of batch normalization

def weight_variable(shape):
	shape = [int(element) for element in shape]
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	shape = [int(element) for element in shape]
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

class ResidualBlock():
	"""Residual Block."""
	def __init__(self, dim_in=256, dim_out=256):
		self.residualBlock1_W = weight_variable([3, 3, dim_in, dim_out])

		self.residualBlock2_W = weight_variable([3, 3, dim_out, dim_out])

	def forward(self, x):
		Y1 = tf.nn.conv2d(x, self.residualBlock1_W, strides=[1, 1, 1, 1], padding='SAME')
		Y1_norm = tf.contrib.layers.batch_norm(Y1)

		Y1_relu = tf.nn.relu(Y1_norm)

		Y2 = tf.nn.conv2d(Y1_relu, self.residualBlock2_W, strides=[1, 1, 1, 1], padding='SAME')
		Y2_norm = tf.contrib.layers.batch_norm(Y2)

		return x + Y2_norm



class Generator():

	def __init__(self, imgDim=(128,128), numClass=5):
		with tf.name_scope("Generator") as self.Gscope:
			self.imgDim = imgDim
			# Weights initialised
			self.downSampling1_W = weight_variable([7, 7, 3 + numClass, 64])
			self.downSampling1_b = bias_variable([64])

			self.downSampling2_W = weight_variable([4, 4, 64, 128])
			self.downSampling2_b = bias_variable([128])

			self.downSampling3_W = weight_variable([4, 4, 128, 256])
			self.downSampling3_b = bias_variable([256])

			self.residualBlock1 = ResidualBlock()
			self.residualBlock2 = ResidualBlock()
			self.residualBlock3 = ResidualBlock()
			self.residualBlock4 = ResidualBlock()
			self.residualBlock5 = ResidualBlock()
			self.residualBlock6 = ResidualBlock()

			self.upSampling1_W = weight_variable([4, 4, 128, 256])

			self.upSampling2_W = weight_variable([4, 4, 64, 128])

			self.upSampling3_W = weight_variable([7, 7, 64, 3])

	def forward(self, X_G):
		with tf.name_scope(self.Gscope):
			batch_size = tf.shape(X_G)[0]

			Y_downSampling1 = tf.nn.relu(tf.nn.conv2d(X_G, self.downSampling1_W, strides=[1, 1, 1, 1], padding='SAME'))
			Y_downSampling1_norm = tf.contrib.layers.batch_norm(Y_downSampling1)

			Y_downSampling2 = tf.nn.relu(tf.nn.conv2d(Y_downSampling1_norm, self.downSampling2_W, strides=[1, 2, 2, 1], padding='SAME'))
			Y_downSampling2_norm = tf.contrib.layers.batch_norm(Y_downSampling2)

			Y_downSampling3 = tf.nn.relu(tf.nn.conv2d(Y_downSampling2_norm, self.downSampling3_W, strides=[1, 2, 2, 1], padding='SAME'))
			Y_downSampling3_norm = tf.contrib.layers.batch_norm(Y_downSampling3)


			Y_residual1 = self.residualBlock1.forward(Y_downSampling3_norm)
			Y_residual2 = self.residualBlock2.forward(Y_residual1)
			Y_residual3 = self.residualBlock3.forward(Y_residual2)
			Y_residual4 = self.residualBlock4.forward(Y_residual3)
			Y_residual5 = self.residualBlock5.forward(Y_residual4)
			Y_residual6 = self.residualBlock6.forward(Y_residual5)


			Y_upSampling1 = tf.nn.relu(tf.nn.conv2d_transpose(Y_residual6, self.upSampling1_W, [batch_size, int(self.imgDim[0]/2), int(self.imgDim[1]/2),128], strides=[1, 2, 2, 1], padding='SAME'))
			Y_upSampling1_norm = tf.contrib.layers.batch_norm(Y_upSampling1)

			Y_upSampling2 = tf.nn.relu(tf.nn.conv2d_transpose(Y_upSampling1_norm, self.upSampling2_W, [batch_size, self.imgDim[0], self.imgDim[1],64], strides=[1, 2, 2, 1], padding='SAME'))
			Y_upSampling2_norm = tf.contrib.layers.batch_norm(Y_upSampling2)

			fakeGeneration = tf.nn.tanh(tf.nn.conv2d(Y_upSampling2_norm, self.upSampling3_W, strides=[1, 1, 1, 1], padding='SAME'))

			return fakeGeneration

	def recForward(self, fake, trueLabels):
		with tf.name_scope(self.Gscope):
			fakeWithRealLabels = tf.concat([fake, trueLabels], 3)
			return self.forward(fakeWithRealLabels)


class Discriminator():

	def __init__(self, imageSize=128, convDim=64, numClass=5):
		with tf.name_scope("Discriminator") as self.Dscope:
			# Weights initialised
			self.imageSize = imageSize
			currDim = convDim

			self.InputLayer = weight_variable([4, 4, 3, currDim])
			self.InputLayer_b = bias_variable([currDim])

			self.HiddenLayer1 = weight_variable([4, 4, currDim, currDim*2])
			self.HiddenLayer1_b = bias_variable([currDim*2])
			currDim = currDim*2

			self.HiddenLayer2 = weight_variable([4, 4, currDim, currDim*2])
			self.HiddenLayer2_b = bias_variable([currDim*2])
			currDim = currDim*2

			self.HiddenLayer3 = weight_variable([4, 4, currDim, currDim*2])
			self.HiddenLayer3_b = bias_variable([currDim*2])
			currDim = currDim*2

			self.HiddenLayer4 = weight_variable([4, 4, currDim, currDim*2])
			self.HiddenLayer4_b = bias_variable([currDim*2])
			currDim = currDim*2

			self.HiddenLayer5 = weight_variable([4, 4, currDim, currDim*2])
			self.HiddenLayer5_b = bias_variable([currDim*2])
			currDim = currDim*2

			self.OutputLayerSrc = weight_variable([3, 3, currDim, 1])

			self.OutputLayerCls = weight_variable([imageSize/64, imageSize/64, currDim, numClass])

	def forward(self, x):
		with tf.name_scope(self.Dscope):
			# x has to be a placeHolder or conexion with generator output
			Y_input = tf.nn.leaky_relu(tf.nn.conv2d(x, self.InputLayer, strides=[1, 2, 2, 1], padding='SAME') + self.InputLayer_b)
			Y_hiddenLayer1 = tf.nn.leaky_relu(tf.nn.conv2d(Y_input, self.HiddenLayer1, strides=[1, 2, 2, 1], padding='SAME') + self.HiddenLayer1_b)
			Y_hiddenLayer2 = tf.nn.leaky_relu(tf.nn.conv2d(Y_hiddenLayer1, self.HiddenLayer2, strides=[1, 2, 2, 1], padding='SAME') + self.HiddenLayer2_b)
			Y_hiddenLayer3 = tf.nn.leaky_relu(tf.nn.conv2d(Y_hiddenLayer2, self.HiddenLayer3, strides=[1, 2, 2, 1], padding='SAME') + self.HiddenLayer3_b)
			Y_hiddenLayer4 = tf.nn.leaky_relu(tf.nn.conv2d(Y_hiddenLayer3, self.HiddenLayer4, strides=[1, 2, 2, 1], padding='SAME') + self.HiddenLayer4_b)
			Y_hiddenLayer5 = tf.nn.leaky_relu(tf.nn.conv2d(Y_hiddenLayer4, self.HiddenLayer5, strides=[1, 2, 2, 1], padding='SAME') + self.HiddenLayer5_b)

			Y_outputLayerSrc = tf.nn.conv2d(Y_hiddenLayer5, self.OutputLayerSrc, strides=[1, 1, 1, 1],padding='SAME')
			Y_outputLayerCls = tf.nn.conv2d(Y_hiddenLayer5, self.OutputLayerCls, strides=[1, 1, 1, 1], padding='VALID') # TODO padding in YCls should BE "TYPE1"

			return Y_outputLayerSrc, Y_outputLayerCls