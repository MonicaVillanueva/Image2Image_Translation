import numpy as np
import tensorflow as tf
import pdb
import scipy.misc as sci
import os
from model import Generator, Discriminator
from ast import literal_eval
from utils import stackLabels, stackLabelsOnly, normalize, denormalize
import random

class Pipeline():
    def __init__(self):
        self.learningRateD = 0.0001
        self.learningRateG = 0.0001
        self.DBpath = 'processed'
        self.graphPath = ''
        self.modelPath = 'model/model.ckpt'
        self.imgDim = 128
        self.batchSize = 16
        self.numClass = 5
        self.lambdaCls = 1
        self.lambdaRec = 10
        self.lambdaGp = 10
        self.g_skip_step = 5
        self.g_skip_count = 1
        self.epochs = 10
        self.epochsDecay = 10
        self.epochsSave = 2
        self.lrDecaysD = np.linspace(self.learningRateD,0,self.epochs-self.epochsDecay+2)
        self.lrDecaysD = self.lrDecaysD[1:]
        self.lrDecaysG = np.linspace(self.learningRateG,0,self.epochs-self.epochsDecay+2)
        self.lrDecaysG = self.lrDecaysG[1:]

    def init_model(self):
        # Create the whole training graph
        self.realX = tf.placeholder(tf.float32, [None, self.imgDim, self.imgDim, 3], name="realX")
        self.realX_fakeLabels = tf.placeholder(tf.float32, [None, self.imgDim, self.imgDim, 3 + self.numClass], name="realX_fakeLabels")

        self.realLabels = tf.placeholder(tf.float32, [None, self.numClass], name="realLabels")
        self.realLabelsOneHot = tf.placeholder(tf.float32, [None, self.imgDim, self.imgDim, self.numClass], name="realLabelsOneHot")
        self.fakeLabels = tf.placeholder(tf.float32, [None, self.numClass], name="fakeLabels")
        self.epsilonph = tf.placeholder(tf.float32, [], name="epsilonph")


        self.lrD = tf.placeholder(tf.float32)
        self.lrG = tf.placeholder(tf.float32)

        # Initialize the generator and discriminator
        self.Gen = Generator()
        self.Dis = Discriminator()

        # Get outputs
        self.fakeX = self.Gen.forward(self.realX_fakeLabels)
        recX = self.Gen.recForward(self.fakeX, self.realLabelsOneHot)
        YSrc_real, self.YCls_real = self.Dis.forward(self.realX)
        YSrc_fake, YCls_fake = self.Dis.forward(self.fakeX)

        self.YCls_real = tf.squeeze(self.YCls_real)  # remove void dimensions
        YCls_fake = tf.squeeze(YCls_fake) # remove void dimensions




        # Create D training pipeline
        self.d_loss_real = tf.reduce_mean(YSrc_real)
        self.d_loss_fake = tf.reduce_mean(YSrc_fake)
        self.d_loss_adv = self.d_loss_real - self.d_loss_fake
        self.d_loss_cls = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.YCls_real,logits=self.realLabels, name="d_loss_cls") / self.batchSize)


        self.d_loss = - self.d_loss_adv + self.lambdaCls * self.d_loss_cls
        #TODO: review parameters

        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('Discriminator/')]
        self.train_D = tf.train.AdamOptimizer(learning_rate=self.lrD, beta1=0.5, beta2=0.999)
        self.train_D_loss = self.train_D.minimize(self.d_loss, var_list=self.d_params)


        #-------------GRADIENT PENALTY---------------------------
        x_hat = self.epsilonph * self.realX + (1.0 - self.epsilonph) * self.fakeX

        # gradient penalty
        YSrc,_ = self.Dis.forward(x_hat)
        gradients = tf.gradients(YSrc, [x_hat])[0]
        gradients_shape = gradients.get_shape().as_list()
        gradients_dim = np.prod(gradients_shape[1:])
        gradients = tf.reshape(gradients, [-1, gradients_dim])
        gradients_norm = tf.reduce_sum(gradients, axis=1)**2
        self._gradient_penalty = tf.reduce_mean(tf.square(gradients_norm - 1.0))# unnecesary mean

        self.d_loss_gp = self.lambdaGp * self._gradient_penalty
        self.train_D_gp = self.train_D.minimize(self.d_loss_gp, var_list=self.d_params)
        #-------------------------------------------------------------------------------

        self.train_D_gradLoss = self.train_D.compute_gradients(self.d_loss, var_list=self.d_params)

        # Create G training pipeline
        g_loss_adv = - tf.reduce_mean(YSrc_fake) # TODO: review this
        g_loss_cls = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=YCls_fake,logits=self.fakeLabels)) / self.batchSize
        g_loss_rec = tf.reduce_mean(tf.abs(self.realX - recX))
        self.g_loss = g_loss_adv + self.lambdaCls * g_loss_cls + self.lambdaRec * g_loss_rec
        self.train_G = tf.train.AdamOptimizer(learning_rate=self.lrG, beta1=0.5, beta2=0.999)
        self.train_G_loss = self.train_G.minimize(self.g_loss)



        #TF session
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.saver = tf.train.Saver()


        #writer = tf.summary.FileWriter(self.graphPath, graph=tf.get_default_graph())

    def train(self):
        # -----------------------------------------------------------------------------------------
        # -----------------------------------START TRAINING----------------------------------------
        # -----------------------------------------------------------------------------------------
        images = []
        trueLabels = []

        for e in range(self.epochs):
            for filename in os.listdir(self.DBpath):
                img = sci.imread(os.path.join(self.DBpath, filename))
                splits = filename.split('_')
                trueLabels.append(literal_eval(splits[1].split('.')[0]))

                # Normalization and random flip (data augmentation)
                img = normalize(img)
                if random.random() > 0.5:
                    img = np.fliplr(img)

                images.append(img)
                if len(images) % self.batchSize == 0:

                    # Create fake labels and associated images
                    randomLabels = np.random.randint(2, size=(self.batchSize, self.numClass))
                    # realX_fakeLabels has to contain the original image with the labels to generate concatenated
                    imagesWithFakeLabels = stackLabels(images, randomLabels)
                    epsilon = np.random.rand()
                    # -----------------------------------------------------------------------------------------
                    # -----------------------------------TRAIN DISCRIMINATOR-----------------------------------
                    # -----------------------------------------------------------------------------------------

                    #print(np.mean(self.sess.run(self.d_params[4])))


                    dloss, _ , _, d_loss_real, d_loss_fake, _gradient_penalty, d_loss_cls, YCls_real, gradLoss = self.sess.run(
                                            [self.d_loss,
                                             self.train_D_loss,
                                             self.train_D_gp,
                                             self.d_loss_real,
                                             self.d_loss_fake,
                                             self._gradient_penalty,
                                             self.d_loss_cls,
                                             self.YCls_real,
                                             self.train_D_gradLoss],
                                            feed_dict={self.lrD: self.learningRateD,
                                                       self.realX_fakeLabels: imagesWithFakeLabels,
                                                       self.realLabels: trueLabels,
                                                       self.realX: np.stack(images),
                                                       self.fakeLabels: randomLabels,
                                                       self.epsilonph: epsilon})



                    if self.g_skip_count == self.g_skip_step:
                        # -----------------------------------------------------------------------------------------
                        # -----------------------------------TRAIN GENERATOR---------------------------------------
                        # -----------------------------------------------------------------------------------------

                        gloss, _ = self.sess.run([self.g_loss, self.train_G_loss],
                                                feed_dict={self.lrG: self.learningRateG,
                                                           self.realX_fakeLabels: imagesWithFakeLabels,
                                                           self.realLabelsOneHot: stackLabelsOnly(trueLabels),
                                                           self.realX: np.stack(images),
                                                           self.fakeLabels: randomLabels})
                        self.g_skip_count = 1
                    else:
                        self.g_skip_count+=1


                    images = []
                    trueLabels = []

                    #print("Loss = " , dloss + gloss, " ", "Dloss = " , dloss, " ", "Gloss = ", gloss, "Epoch =", e)
                    print("Dloss = " , dloss, " d_loss_real: ", d_loss_real, " d_loss_fake: ", d_loss_fake, " gradient penalty: ", _gradient_penalty, " d_loss_cls: ", d_loss_cls)
                    #print("YCls_real: ")
                    #print(YCls_real)
                    #print([np.mean(i) for i in gradLoss])

            if (e+1) >= self.epochsDecay:
                self.lrD = self.lrDecaysD[0]
                self.lrG = self.lrDecaysG[0]
                self.lrDecaysD = self.lrDecaysD[1:]
                self.lrDecaysG = self.lrDecaysG[1:]


            # Save model every epochsSave epochs
            if e == 0:
                if not os.path.exists(self.modelPath):
                    os.makedirs(self.modelPath)

            elif e % self.epochsSave == 0:
                self.saver.save(self.sess, self.modelPath)



    def test(self, labels=None, img=None):
        if not os.path.exists(self.modelPath):
            print("The model does not exit")
            return

        with tf.Session() as sess:
            # Restore model weights from previously saved model
            folder = os.path.dirname(os.path.normpath(self.modelPath))
            saver = tf.train.import_meta_graph(os.path.join(folder,'model.ckpt.meta'))
            saver.restore(sess, tf.train.latest_checkpoint(folder))

            if img is None:
                img = sci.imread(os.path.join(self.DBpath, "000001_[0, 0, 1, 0, 1].jpg"))
            if labels is None:
                labels = np.random.randint(2, size=(1, 5))

            testImage = stackLabels([img], labels)
            generatedImage = np.squeeze(self.sess.run([self.fakeX], feed_dict={self.realX_fakeLabels: testImage}))

            # sci.imsave('img.jpg', img)
            # img = normalize(img)
            # sci.imsave('out.jpg', denormalize(img))

            sci.imsave('outfile1.jpg', denormalize(generatedImage))

