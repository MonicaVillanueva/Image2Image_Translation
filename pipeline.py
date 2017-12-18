import numpy as np
import tensorflow as tf
import pdb
import scipy.misc as sci
import os
from model import Generator, Discriminator
from ast import literal_eval
from utils import stackLabels, stackLabelsOnly


class Pipeline():
    def __init__(self):
        self.learningRateD = 0.1
        self.learningRateG = 0.0001
        self.DBpath = 'processed'
        self.imgDim = 128
        self.batchSize = 16
        self.numClass = 5
        self.lambdaCls = 1
        self.lambdaRec = 10

    def init_model(self):
        # Create the whole training graph
        self.realX = tf.placeholder(tf.float32, [None, self.imgDim, self.imgDim, 3], name="realX")
        self.realX_fakeLabels = tf.placeholder(tf.float32, [None, self.imgDim, self.imgDim, 3 + self.numClass], name="realX_fakeLabels")

        self.realLabels = tf.placeholder(tf.float32, [None, 1, 1, self.numClass], name="realLabels") # same size as YCls_real
        self.realLabelsOneHot = tf.placeholder(tf.float32, [None, self.imgDim, self.imgDim, self.numClass], name="realLabelsOneHot")
        self.fakeLabels = tf.placeholder(tf.float32, [None, 1, 1, self.numClass], name="fakeLabels")

        self.lrD = tf.placeholder(tf.float32)
        self.lrG = tf.placeholder(tf.float32)

        # Initialize the generator and discriminator
        self.Gen = Generator()
        self.Dis = Discriminator()

        # Get outputs
        self.fakeX = self.Gen.forward(self.realX_fakeLabels)
        recX = self.Gen.recForward(self.fakeX, self.realLabelsOneHot)
        YSrc_real, YCls_real = self.Dis.forward(self.realX)
        YSrc_fake, YCls_fake = self.Dis.forward(self.fakeX)



        # Create D training pipeline
        d_loss_real = tf.reduce_mean(YSrc_real)
        d_loss_fake = tf.reduce_mean(YSrc_fake)
        d_loss_adv = d_loss_real - d_loss_fake

        d_loss_cls = tf.nn.sigmoid_cross_entropy_with_logits(labels=YCls_real,logits=self.realLabels, name="d_loss_cls")

        d_loss = - d_loss_adv + self.lambdaCls * d_loss_cls

        #TODO: add D train step
        # self.train_D = tf.train.AdamOptimizer(




        # Create D training pipeline
        g_loss_adv = - tf.reduce_mean(YSrc_fake) # TODO: review this
        g_loss_cls = tf.nn.sigmoid_cross_entropy_with_logits(labels=YCls_fake,logits=self.fakeLabels) / self.batchSize
        g_loss_rec = tf.reduce_mean(tf.abs(self.realX - recX))

        self.g_loss = g_loss_adv + self.lambdaCls * g_loss_cls + self.lambdaRec * g_loss_rec

        self.train_G = tf.train.AdamOptimizer(learning_rate=self.lrG, beta1=0.5, beta2=0.999).minimize(self.g_loss)


        #TF session
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def train(self):
        # -----------------------------------------------------------------------------------------
        # -----------------------------------START TRAINING----------------------------------------
        # -----------------------------------------------------------------------------------------
        images = []
        trueLabels = []

        # TODO: for n batches
        for filename in os.listdir(self.DBpath):
            img = sci.imread(os.path.join(self.DBpath, filename))
            splits = filename.split('_')
            trueLabels.append(literal_eval(splits[1].split('.')[0]))
            images.append(img)
            if len(images) % self.batchSize == 0:
                # TRAIN GENERATOR
                # -------------------------------------------------------------------------------------
                # TODO: only each 5th time
                # X_G has to contain the original image with the labels to generate concatenated
                randomLabels = np.random.randint(2, size=(self.batchSize, self.numClass))
                imagesWithFakeLabels = stackLabels(images, randomLabels)

                # Reformat randomLabels to fit feeder
                randomLabels = np.expand_dims(randomLabels, axis=1)
                randomLabels = np.expand_dims(randomLabels, axis=1)

                loss, _ = self.sess.run([self.g_loss, self.train_G],
                                        feed_dict={self.lrG: self.learningRateG,
                                                   self.realX_fakeLabels: imagesWithFakeLabels,
                                                   self.realLabelsOneHot: stackLabelsOnly(trueLabels),
                                                   self.realX: np.stack(images),
                                                   self.fakeLabels: randomLabels})

                pdb.set_trace()
                images = []
                trueLabels = []

    def test(self):
        pass

p = Pipeline()
p.init_model()
p.train()