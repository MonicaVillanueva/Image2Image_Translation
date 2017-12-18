import numpy as np
import tensorflow as tf
import pdb
import scipy.misc as sci
import os
from model import Generator, Discriminator
from ast import literal_eval
from utils import stackLabels
from model import weight_variable, bias_variable
class Pipeline():
    def __init__(self, config):
        self.param1 = config.parameter1
        self.param2 = config.parameter2
        self.lr = 0.1
        self.DBpath = 'D:/processed'
        self.imgDim = 128
        self.batchSize = 16
        self.numClass = 5

    def Discriminator(x, imageSize=128, convDim=64, numClass=5):
        # Weights initialised
        with tf.name_scope('Discriminator') as Dscope:
            imageSize = imageSize
            currDim = convDim

            x = tf.variable_scope

            InputLayer = weight_variable([4, 4, 3, currDim])
            InputLayer_b = bias_variable([currDim])

            HiddenLayer1 = weight_variable([4, 4, currDim, currDim * 2])
            HiddenLayer1_b = bias_variable([currDim * 2])
            currDim = currDim * 2

            HiddenLayer2 = weight_variable([4, 4, currDim, currDim * 2])
            HiddenLayer2_b = bias_variable([currDim * 2])
            currDim = currDim * 2

            HiddenLayer3 = weight_variable([4, 4, currDim, currDim * 2])
            HiddenLayer3_b = bias_variable([currDim * 2])
            currDim = currDim * 2

            HiddenLayer4 = weight_variable([4, 4, currDim, currDim * 2])
            HiddenLayer4_b = bias_variable([currDim * 2])
            currDim = currDim * 2

            HiddenLayer5 = weight_variable([4, 4, currDim, currDim * 2])
            HiddenLayer5_b = bias_variable([currDim * 2])
            currDim = currDim * 2

            OutputLayerSrc = weight_variable([3, 3, currDim, 1])
            OutputLayerSrc_b = bias_variable([1])

            OutputLayerCls = weight_variable([imageSize / 64, imageSize / 64, currDim, numClass])
            OutputLayerCls_b = bias_variable([numClass])

            # x has to be a placeHolder or conexion with generator output
            Y_input = tf.nn.relu(tf.nn.conv2d(x, InputLayer, strides=[1, 2, 2, 1], padding='SAME') + InputLayer_b)
            Y_hiddenLayer1 = tf.nn.relu(tf.nn.conv2d(Y_input, HiddenLayer1, strides=[1, 2, 2, 1], padding='SAME') + HiddenLayer1_b)
            Y_hiddenLayer2 = tf.nn.relu(tf.nn.conv2d(Y_hiddenLayer1, HiddenLayer2, strides=[1, 2, 2, 1],padding='SAME') + HiddenLayer2_b)
            Y_hiddenLayer3 = tf.nn.relu(tf.nn.conv2d(Y_hiddenLayer2, HiddenLayer3, strides=[1, 2, 2, 1],padding='SAME') + HiddenLayer3_b)
            Y_hiddenLayer4 = tf.nn.relu(tf.nn.conv2d(Y_hiddenLayer3, HiddenLayer4, strides=[1, 2, 2, 1], padding='SAME') + HiddenLayer4_b)
            Y_hiddenLayer5 = tf.nn.relu(tf.nn.conv2d(Y_hiddenLayer4, HiddenLayer5, strides=[1, 2, 2, 1],padding='SAME') + HiddenLayer5_b)

            Y_outputLayerSrc = tf.nn.conv2d(Y_hiddenLayer5, OutputLayerSrc, strides=[1, 1, 1, 1],padding='SAME') + OutputLayerSrc_b
            Y_outputLayerCls = tf.nn.conv2d(Y_hiddenLayer5, OutputLayerCls, strides=[1, 1, 1, 1],padding='VALID') + OutputLayerCls_b  # TODO padding in YCls should BE "TYPE1"

            return Y_outputLayerSrc,Y_outputLayerCls

    def init_model(self):

        # Initialize optimizers
        #self.train_G = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.999)
        #self.train_D = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.999)

        #create the whole training graph
        self.realX = tf.placeholder(tf.float32, [None, self.imgDim, self.imgDim, 3], name="realX")
        self.realX_fakeLabels = tf.placeholder(tf.float32, [None, self.imgDim, self.imgDim, 3 + self.numClass], name="realX_fakeLabels")
        self.realLabels = tf.placeholder(tf.float32, [None, 1, 1, self.numClass], name="realLabels")

        # Initilize the generator and discriminator
        self.Gen = Generator()
        self.Dis = Discriminator()

        #create D training pipeline
        YSrc1, YCls1 =self.Dis.forward(self.realX)
        d_loss_real = tf.reduce_mean(YSrc1)


        fakeX = self.Gen.forward(self.realX_fakeLabels)
        YSrc2, YCls2 =self.Dis.forward(fakeX)
        d_loss_fake = tf.reduce_mean(YSrc2)


        d_loss_cls = tf.nn.softmax_cross_entropy_with_logits(labels=YCls1,logits=self.realLabels, name="d_loss_cls")

        d_loss = d_loss_real + d_loss_fake + d_loss_cls




        #TF session
        self.init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def train(self):
        # -----------------------------------------------------------------------------------------
        # -----------------------------------START TRAINING----------------------------------------
        # -----------------------------------------------------------------------------------------
        images = []
        trueLabels = []

        for filename in os.listdir(self.DBpath):
            # ??? Normalize images??
            img = sci.imread(os.path.join(self.DBpath, filename))
            splits = filename.split('_')
            trueLabels.append(literal_eval(splits[1].split('.')[0]))
            images.append(img)
            if len(images) % self.batchSize == 0:
                # TRAIN GENERATOR
                # -------------------------------------------------------------------------------------
                # TODO: only each 5th time
                # X_G has to contain the original image with the labels to generate concatenated
                imagesWithFakeLabels = stackLabels(images, np.random.randint(2, size=(self.batchSize, self.numClass)))
                fake = sess.run(fakeGeneration, feed_dict={imagesWithFakeLabelsT: imagesWithFakeLabels})
                fakeWithRealLabels = stackLabels(fake, trueLabels)

                loss, _ = sess.run([gLoss, train_G],
                                   feed_dict={lr: learningRate, fakeWithRealLabelsT: fakeWithRealLabels,
                                              imagesWithFakeLabelsT: imagesWithFakeLabels, real: np.stack(images)})

                pdb.set_trace()
                images = []
                trueLabels = []

    def test(self):
        pass
