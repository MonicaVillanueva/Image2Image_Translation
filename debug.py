import numpy as np
import pdb
import scipy.misc as sci
import os
from model import Generator
from ast import literal_eval

path = "processed"

imgDim = 128
batchSize = 16
numClass = 5

G = Generator()

images = []

for filename in os.listdir(path):
	# ??? Normalize images??
	img = sci.imread(os.path.join(path, filename))
	splits = filename.split('_')
	labels = literal_eval(splits[1].split('.')[0])
	for l in labels:
		label = l * np.ones([imgDim,imgDim])
		img = np.dstack([img, label])
	images.append(img)
	if len(images) % batchSize == 0:
		fakeLabels = np.random.randint(2, size=(batchSize, numClass))
		fake = G.forward(images, fakeLabels)
		pdb.set_trace()
		images = []
		pdb.set_trace()