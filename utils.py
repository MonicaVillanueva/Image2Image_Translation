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

def normalize(img):
	img = img.astype('float64')
	img = (img - img.min()) / (img.max() - img.min())
	img = (img - 0.5) / 0.5
	return img

def denormalize(img):
	img = (img + 1) / 2
	return np.clip(img, 0, 1)

def normalize2(img):
	img = (img - 0.5) / 0.5
	return img

def denormalize2(img):
	img = (img + 1) / 2
	return img