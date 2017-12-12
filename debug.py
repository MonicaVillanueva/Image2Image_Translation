import tensorflow as tf
import numpy as np
import pdb
import scipy.misc as sci
import os
from model import Generator

path = "processed"

imgDim = 128
batchSize = 16

with tf.Session() as sess:
	tf.initialize_all_variables().run()

	G = Generator()

	images = []

	for filename in os.listdir(path):
		img = sci.imread(os.path.join(path, filename))
		images.append(img)
		if len(images) % batchSize == 0:
			# TODO: add labels
			# ??? Normalize images??
			print(sess.run([G.train()], feed_dict={X: np.stack(images)}))
			images = []
			pdb.set_trace()