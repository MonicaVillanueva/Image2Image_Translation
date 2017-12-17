import tensorflow as tf
import argparse
import model
# from PIL import Image
import numpy as np
from scipy.misc import imread, imresize, imsave
import os



def data_input_fn(path, filenames, batch_size=16, shuffle=False, repeat=None):
    def parser(serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
            })

        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        depth = tf.cast(features['depth'], tf.int32)
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        label = tf.decode_raw(features['label'], tf.uint8)
        
        image = tf.reshape(image, (height, width, depth))

        # Normalize the values of the image from the range [0, 255] to [-0.5, 0.5] #FIXME: normalize or don't
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

        return image, label


    # Import data
    list = [os.path.join(path,f) for f in filenames]
    dataset = tf.data.TFRecordDataset(list) # tf.contrib.data.TFRecordDataset(list)

    # Map the parser over dataset, and batch results by up to batch_size
    dataset = dataset.map(parser, num_parallel_calls=1)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=128)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(repeat)
    iterator = dataset.make_one_shot_iterator()

    # features, labels = iterator.get_next()

    return iterator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    imageSize = 128

    # Model parameters
    parser.add_argument('--parameter1', type=int, default=0)
    parser.add_argument('--parameter2', type=float, default=0)

    config = parser.parse_args()
    print(config)

    # Discriminator testing
    print(tf.__version__)
    train_filenames = os.listdir(os.getcwd() + '\\records')
    iterator = data_input_fn(os.getcwd() + '\\records', train_filenames[1:3], batch_size=16)
    features, labels = iterator.get_next()

    # img1 = np.array(imread("processed/000001_[0, 0, 1, 0, 1].jpg", mode='RGB')).reshape(1,128,128,3)
    # img2 = np.array(imread("processed/000002_[0, 0, 1, 0, 1].jpg", mode='RGB')).reshape(1,128,128,3)
    # batch = np.append(img1,img2, axis=0)

    print(features.shape)
    Disc = model.Discriminator()
    # X_D = tf.placeholder(tf.float32, shape=[None, imageSize, imageSize, 3])
    Y_outputLayerSrc, Y_outputLayerCls = Disc.forward(features)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    YSrc, YCls = sess.run([Y_outputLayerSrc, Y_outputLayerCls])
    YSrc, YCls = YSrc.squeeze(), YCls.squeeze()

    print("YSrc: ", YSrc.shape)
    print("YCls: ", YCls.shape)