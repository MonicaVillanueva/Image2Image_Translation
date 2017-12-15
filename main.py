import tensorflow as tf
import argparse
import model
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize, imsave

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    imageSize = 128

    # Model parameters
    parser.add_argument('--parameter1', type=int, default=0)
    parser.add_argument('--parameter2', type=float, default=0)

    config = parser.parse_args()
    print(config)

    # Discriminator testing
    img1 = np.array(imread("processed/000001_[0, 0, 1, 0, 1].jpg", mode='RGB')).reshape(1,128,128,3)
    img2 = np.array(imread("processed/000002_[0, 0, 1, 0, 1].jpg", mode='RGB')).reshape(1,128,128,3)
    batch = np.append(img1,img2, axis=0)

    print(batch.shape)
    Disc = model.Discriminator()
    X_D = tf.placeholder(tf.float32, shape=[None, imageSize, imageSize, 3])
    Y_outputLayerSrc, Y_outputLayerCls = Disc.forward(X_D)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    YSrc, YCls = sess.run([Y_outputLayerSrc, Y_outputLayerCls], feed_dict={X_D: batch})
    YSrc, YCls = YSrc.squeeze(), YCls.squeeze()

    print("YSrc: ", YSrc.shape)
    print("YCls: ", YCls.shape)