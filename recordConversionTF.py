import tensorflow as tf
import os
import ast
import scipy.misc as sci
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt


db_path = os.getcwd() + '\processed'
output_path = 'C:\\Users\\python\\gen_images'



def extract_info(paths):

    images = np.empty((len(paths), 128, 128, 3), dtype=np.uint8)
    labels = np.empty((len(paths), 5), dtype=np.int32)

    cont = 0
    for filename in paths:
        splits = filename.split('_')
        name = splits[0]
        label = splits[1].split('.')[0]

        images[cont, :, :] = plt.imread(os.path.join(db_path, filename))
        # plt.imshow(images[0])
        labels[cont, :] = np.asarray(ast.literal_eval(label))

        cont += 1

    return [images, labels]


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(list, name):    #(images, labels, name):

    images, labels = extract_info(list)

    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                         (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(output_path, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        labels_raw = labels[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _bytes_feature(labels_raw),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


def main(argv):

    list_imgs = os.listdir(db_path)
    shuffle(list_imgs)

    train_list_imgs = list_imgs[0:len(list_imgs)-2000]
    test_list_imgs = list_imgs[len(list_imgs)-2000:]

    # shards = 100000 # 40
    num_files_shard = 4800 # 16(batch_size) * 3
    shards = round(len(train_list_imgs)/num_files_shard)

    for s in range(shards):
        init = s*num_files_shard
        end = min((s+1)*num_files_shard-1, len(train_list_imgs)-1)
        imgs = train_list_imgs[init:end]
        print('printing shard %d of %d' % (s, shards))
        convert_to(imgs, 'train-%d-of-%d' % (s, shards))

    convert_to(test_list_imgs, 'test')


if __name__ == '__main__':
    tf.app.run()