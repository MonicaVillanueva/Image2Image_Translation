import numpy as np
import tensorflow as tf
import pdb
import scipy.misc as sci
import os
from model import Generator, Discriminator
from ast import literal_eval
from utils import stackLabels

path = "processed"

imgDim = 128
batchSize = 16
numClass = 5

learningRate = 0.0001


# -----------------------------------------------------------------------------------------
# -----------------------------------CREATE GRAPH------------------------------------------
# -----------------------------------------------------------------------------------------

G = Generator()
D = Discriminator()

# Generator Losses
imagesWithFakeLabelsT = tf.placeholder(tf.float32, shape=[None, imgDim, imgDim, 3 + numClass])
fakeWithRealLabelsT = tf.placeholder(tf.float32, shape=[None, imgDim, imgDim, 3 + numClass])
fakeGeneration = G.forward(imagesWithFakeLabelsT)
recGeneration = G.forward(fakeWithRealLabelsT)
real = tf.placeholder(tf.float32, shape=[None, imgDim, imgDim, 3])
recLossG = tf.reduce_mean(tf.abs(real - recGeneration))

Y_outputLayerSrc_Fake, Y_outputLayerCls_Fake = D.forward(fakeGeneration)
# TODO: Dcls_Fake = ...
Dadv_fake = - tf.reduce_mean(Y_outputLayerSrc_Fake)

gLoss = recLossG + Dadv_fake #+ Dcls_Fake

# Discrimintar losses
# X_D = tf.placeholder(tf.float32, shape=[None, imageSize, imageSize, 3])
# Y_outputLayerSrc_Real, Y_outputLayerCls_Real = Disc.forward(X_D)
# dLoss = G.fakeImage

# Train
lr = tf.placeholder(tf.float32)

# TODO: add decay
train_G = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.999).minimize(gLoss)
# train_D = tf.train.AdamOptimizer(...........).minimize(self.dLoss)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


# -----------------------------------------------------------------------------------------
# -----------------------------------START TRAINING----------------------------------------
# -----------------------------------------------------------------------------------------


images = []
trueLabels = []

for filename in os.listdir(path):
	# ??? Normalize images??
	img = sci.imread(os.path.join(path, filename))
	splits = filename.split('_')
	trueLabels.append(literal_eval(splits[1].split('.')[0]))
	images.append(img)
	if len(images) % batchSize == 0:
		# TRAIN GENERATOR
		# -------------------------------------------------------------------------------------
		# TODO: only each 5th time
		# X_G has to contain the original image with the labels to generate concatenated
		imagesWithFakeLabels = stackLabels(images, np.random.randint(2, size=(batchSize, numClass)))
		fake = sess.run(fakeGeneration, feed_dict={imagesWithFakeLabelsT: imagesWithFakeLabels})
		fakeWithRealLabels = stackLabels(fake, trueLabels)

		loss, _ = sess.run([gLoss, train_G], feed_dict={lr: learningRate, fakeWithRealLabelsT: fakeWithRealLabels, imagesWithFakeLabelsT: imagesWithFakeLabels, real: np.stack(images)})


		pdb.set_trace()
		images = []
		trueLabels = []
