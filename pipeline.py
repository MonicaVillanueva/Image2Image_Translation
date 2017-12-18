import numpy as np
import tensorflow as tf
import pdb
import scipy.misc as sci
import os
from model import Generator, Discriminator
from ast import literal_eval
from utils import stackLabels

class Pipeline():
    def __init__(self):
        self.learningRateD = 0.1
        self.learningRateG = 0.0001
        self.DBpath = 'processed'
        self.imgDim = 128
        self.batchSize = 16
        self.numClass = 5

    def init_model(self):
        #create the whole training graph
        self.realX = tf.placeholder(tf.float32, [None, self.imgDim, self.imgDim, 3], name="realX")
        self.realX_fakeLabels = tf.placeholder(tf.float32, [None, self.imgDim, self.imgDim, 3 + self.numClass], name="realX_fakeLabels")
        self.fakeX_realLabels = tf.placeholder(tf.float32, shape=[None, self.imgDim, self.imgDim, 3 + self.numClass], name="fakeX_realLabels")
        self.realLabels = tf.placeholder(tf.float32, [None, 1, self.numClass], name="realLabels")
        self.lrD = tf.placeholder(tf.float32)
        self.lrG = tf.placeholder(tf.float32)

        # Initilize the generator and discriminator
        self.Gen = Generator()
        self.Dis = Discriminator()

        # Get outputs
        self.fakeX = self.Gen.forward(self.realX_fakeLabels)
        recX = self.Gen.forward(self.fakeX_realLabels)
        YSrc_real, YCls_real =self.Dis.forward(self.realX)
        YSrc_fake, YCls_fake =self.Dis.forward(self.fakeX)



        # Create D training pipeline
        
        d_loss_real = tf.reduce_mean(YSrc_real)
        
        d_loss_fake = tf.reduce_mean(YSrc_fake)

        d_loss_cls = tf.nn.softmax_cross_entropy_with_logits(labels=YCls_real,logits=self.realLabels, name="d_loss_cls")

        d_loss = d_loss_real + d_loss_fake + d_loss_cls

        #TODO: add D train step
        # self.train_D = tf.train.AdamOptimizer(

        # Create D training pipeline
        recLossG = tf.reduce_mean(tf.abs(self.realX - recX))

        # TODO: Dcls_Fake = ...

        # TODO: review this
        Dadv_fake = - tf.reduce_mean(YSrc_fake)

        self.gLoss = recLossG + Dadv_fake #+ Dcls_Fake

        self.train_G = tf.train.AdamOptimizer(learning_rate=self.lrG, beta1=0.5, beta2=0.999).minimize(self.gLoss)


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
                imagesWithFakeLabels = stackLabels(images, np.random.randint(2, size=(self.batchSize, self.numClass)))
                fake = self.sess.run(self.fakeX, feed_dict={self.realX_fakeLabels: imagesWithFakeLabels})
                fakeWithRealLabels = stackLabels(fake, trueLabels)

                loss, _ = self.sess.run([self.gLoss, self.train_G],
                                   feed_dict={self.lrG: self.learningRateG, self.fakeX_realLabels: fakeWithRealLabels,
                                              self.realX_fakeLabels: imagesWithFakeLabels, self.realX: np.stack(images)})

                pdb.set_trace()
                images = []
                trueLabels = []

    def test(self):
        pass

p = Pipeline()
p.init_model()
p.train()