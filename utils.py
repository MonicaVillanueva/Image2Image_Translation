import numpy as np

def stackLabels(imgs, labels):
	result = []
	for idx, item in enumerate(imgs):
		result.append(_stackLabels(item,labels[idx]))

	return np.stack(result)


def _stackLabels(img, labels, imgDim=128):
	for l in labels:
		label = l * np.ones([imgDim,imgDim])
		img = np.dstack([img, label])

	return img

def stackLabelsOnly(labels):
	result = []
	for label in labels:
		result.append(_stackLabelsOnly(label))

	return np.stack(result)

def _stackLabelsOnly(labels, imgDim=128):
	result = []
	for l in labels:
		result.append(l * np.ones([imgDim,imgDim]))

	return np.dstack(result)