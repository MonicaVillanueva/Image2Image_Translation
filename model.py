import tensorflow as tf
import numpy as np

def _instance_norm(net, train=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

class ResidualBlock():
	"""Residual Block."""
	def __init__(self, dim_in=256, dim_out=256):
		self.Y1 = tf.layers.Conv2D(filters=dim_out, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,use_bias=False, strides=1)
		#self.Y1_norm = tf.layers.BatchNormalization()
		self.Y2 = tf.layers.Conv2D(filters=dim_out, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,use_bias=False, strides=1)
		#self.Y2_norm = tf.layers.BatchNormalization()

	def forward(self, x):
		h = self.Y1(x)
		h = _instance_norm(h)
		h = tf.nn.relu(h)
		h = self.Y2(h)
		h = _instance_norm(h)

		return x + h

class Generator():

	def __init__(self, imgDim=(128,128), numClass=5):
		with tf.variable_scope("Generator") as self.Gscope:
			self.imgDim = imgDim

			# FIXME PADDING P3...EN TF?
			self.downSampling1 = tf.layers.Conv2D(filters=64,kernel_size=[7, 7],padding="same",activation=tf.nn.relu, use_bias=False, strides=1)
			#self.downSampling1_norm = tf.layers.BatchNormalization()

			self.downSampling2 = tf.layers.Conv2D(filters=128,kernel_size=[4, 4],padding="same",activation=tf.nn.relu, use_bias=False, strides=2)
			#self.downSampling2_norm = tf.layers.BatchNormalization()

			self.downSampling3 = tf.layers.Conv2D(filters=256,kernel_size=[4, 4],padding="same",activation=tf.nn.relu, use_bias=False, strides=2)
			#self.downSampling3_norm = tf.layers.BatchNormalization()

			self.residualBlock1 = ResidualBlock()
			self.residualBlock2 = ResidualBlock()
			self.residualBlock3 = ResidualBlock()
			self.residualBlock4 = ResidualBlock()
			self.residualBlock5 = ResidualBlock()
			self.residualBlock6 = ResidualBlock()

			self.Y_upSampling1 = tf.layers.Conv2DTranspose(filters=128,kernel_size=[4, 4],padding="same",activation=tf.nn.relu, use_bias=False, strides=2)
			#self.Y_upSampling1_norm = tf.layers.BatchNormalization()
			self.Y_upSampling2 = tf.layers.Conv2DTranspose(filters=64,kernel_size=[4, 4],padding="same",activation=tf.nn.relu, use_bias=False, strides=2)
			#self.Y_upSampling2_norm = tf.layers.BatchNormalization()

			self.fakeGeneration = tf.layers.Conv2D(filters=3,kernel_size=[7, 7],padding="same",activation=tf.nn.tanh, use_bias=False, strides=1) #FIXME PADDING P3 IN TF...?



	def forward(self, X_G):
		with tf.variable_scope(self.Gscope):
			batch_size = tf.shape(X_G)[0]

			h = self.downSampling1(X_G)
			h = _instance_norm(h)
			h = self.downSampling2(h)
			h = _instance_norm(h)
			h = self.downSampling3(h)
			h = _instance_norm(h)


			h = self.residualBlock1.forward(h)
			h = self.residualBlock2.forward(h)
			h = self.residualBlock3.forward(h)
			h = self.residualBlock4.forward(h)
			h = self.residualBlock5.forward(h)
			h = self.residualBlock6.forward(h)

			h = self.Y_upSampling1(h)
			h = _instance_norm(h)
			h = self.Y_upSampling2(h)
			h = _instance_norm(h)

			return self.fakeGeneration(h)

	def recForward(self, fake, trueLabels):
		with tf.variable_scope(self.Gscope):
			fakeWithRealLabels = tf.concat([fake, trueLabels], 3)
			return self.forward(fakeWithRealLabels)


class Discriminator():

	def __init__(self, imageSize=128, convDim=64, numClass=5):
		with tf.variable_scope("Discriminator") as self.Dscope:
			# Weights initialised
			self.imageSize = imageSize
			currDim = convDim

			self.InputLayer   = tf.layers.Conv2D(filters=currDim,kernel_size=[4, 4],padding="same",activation=tf.nn.leaky_relu, use_bias=True, strides=2)

			self.HiddenLayer1 = tf.layers.Conv2D(filters=currDim*2,kernel_size=[4, 4],padding="same",activation=tf.nn.leaky_relu, use_bias=True, strides=2)
			currDim = currDim*2
			self.HiddenLayer2 = tf.layers.Conv2D(filters=currDim*2,kernel_size=[4, 4],padding="same",activation=tf.nn.leaky_relu, use_bias=True, strides=2)
			currDim = currDim*2
			self.HiddenLayer3 = tf.layers.Conv2D(filters=currDim*2,kernel_size=[4, 4],padding="same",activation=tf.nn.leaky_relu, use_bias=True, strides=2)
			currDim = currDim*2
			self.HiddenLayer4 = tf.layers.Conv2D(filters=currDim*2,kernel_size=[4, 4],padding="same",activation=tf.nn.leaky_relu, use_bias=True, strides=2)
			currDim = currDim*2
			self.HiddenLayer5 = tf.layers.Conv2D(filters=currDim*2,kernel_size=[4, 4],padding="same",activation=tf.nn.leaky_relu, use_bias=True, strides=2)

			self.OutputLayerSrc = tf.layers.Conv2D(filters=1,kernel_size=[3, 3],padding="same",activation=None, use_bias=False, strides=1)

			self.OutputLayerCls = tf.layers.Conv2D(filters=numClass,kernel_size=[imageSize/64, imageSize/64],padding="valid",activation=None, use_bias=False, strides=1)


	def forward(self, x):
		with tf.variable_scope(self.Dscope):

			h = self.InputLayer(x)
			h = self.HiddenLayer1(h)
			h = self.HiddenLayer2(h)
			h = self.HiddenLayer3(h)
			h = self.HiddenLayer4(h)
			h = self.HiddenLayer5(h)

			YSrc = self.OutputLayerSrc(h)
			YCls = self.OutputLayerCls(h)

			return YSrc,YCls