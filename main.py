
import argparse
import model
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize, imsave

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--parameter1', type=int, default=0)
    parser.add_argument('--parameter2', type=float, default=0)

    config = parser.parse_args()
    print(config)

    # Discriminator testing
    img1 = np.array(imread("D:/processed/000001_[0, 0, 1, 0, 1].jpg", mode='RGB')).reshape(1,128,128,3)
    img2 = np.array(imread("D:/processed/000002_[0, 0, 1, 0, 1].jpg", mode='RGB')).reshape(1,128,128,3)
    batch = np.append(img1,img2, axis=0)

    print(batch.shape)
    Disc = model.Discriminator()
    YSrc, YCls = Disc.forward(batch)

    print("YSrc: ", YSrc.shape)
    print("YCls: ", YCls.shape)