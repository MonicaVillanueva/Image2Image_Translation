import numpy as np
import tensorflow as tf
import pdb
import scipy.misc as sci
import os
from model import Generator, Discriminator
from ast import literal_eval
from utils import stackLabels, stackLabelsOnly, normalize, denormalize
import random
import scipy.misc



def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)

class Pipeline():
    def __init__(self):
        self.learningRateD = 0.0001
        self.learningRateG = 0.0001
        self.DBpath = 'D:/processed/'
        self.graphPath = ''
        self.modelPath = 'model/model.ckpt'
        self.imgDim = 128
        self.batchSize = 4
        self.numClass = 5
        self.lambdaCls = 1
        self.lambdaRec = 10
        self.lambdaGp = 10
        self.g_skip_step = 10
        self.epochs = 10
        self.epochsDecay = 10
        self.epochsSave = 2
        self.lrDecaysD = np.linspace(self.learningRateD,0,self.epochs-self.epochsDecay+2)
        self.lrDecaysD = self.lrDecaysD[1:]
        self.lrDecaysG = np.linspace(self.learningRateG,0,self.epochs-self.epochsDecay+2)
        self.lrDecaysG = self.lrDecaysG[1:]
        self.clipD = 0.05

    def init_model(self):
        # Create the whole training graph
        self.realX = tf.placeholder(tf.float32, [None, self.imgDim, self.imgDim, 3], name="realX")
        self.realLabels = tf.placeholder(tf.float32, [None, self.numClass], name="realLabels")
        self.realLabelsOneHot = tf.placeholder(tf.float32, [None, self.imgDim, self.imgDim, self.numClass], name="realLabelsOneHot")
        self.fakeLabels = tf.placeholder(tf.float32, [None, self.numClass], name="fakeLabels")
        self.fakeLabelsOneHot = tf.placeholder(tf.float32, [None, self.imgDim, self.imgDim, self.numClass], name="fakeLabelsOneHot")
        self.alphagp = tf.placeholder(tf.float32, [], name="alphagp")


        # Initialize the generator and discriminator
        self.Gen = Generator()
        self.Dis = Discriminator()



        # -----------------------------------------------------------------------------------------
        # -----------------------------------Create D training pipeline----------------------------
        # -----------------------------------------------------------------------------------------

        # Create fake image
        self.fakeX = self.Gen.recForward(self.realX, self.fakeLabelsOneHot)
        YSrc_real, YCls_real = self.Dis.forward(self.realX)
        YSrc_fake, YCls_fake = self.Dis.forward(self.fakeX)

        YCls_real = tf.squeeze(YCls_real)  # remove void dimensions
        self.d_loss_real = - tf.reduce_mean(YSrc_real)
        self.d_loss_fake = tf.reduce_mean(YSrc_fake)
        self.d_loss_cls = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.realLabels,logits=YCls_real, name="d_loss_cls")) / self.batchSize


        #-------------GRADIENT PENALTY---------------------------
        interpolates = self.alphagp * self.realX + (1 - self.alphagp) * self.fakeX

        gradients = tf.gradients(self.Dis.forward(interpolates)[0], [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
        _gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0))
        self.d_loss_gp = self.lambdaGp * _gradient_penalty
        #self.train_D_gp = train_D.minimize(self.d_loss_gp, var_list=self.d_params)
        # gvs = self.train_D.compute_gradients(self.d_loss_gp)
        # capped_gvs = [(ClipIfNotNone(grad), var) for grad, var in gvs]
        # self.train_D_gp = self.train_D.apply_gradients(capped_gvs)
        #-------------------------------------------------------------------------------

        #TOTAL LOSS
        self.d_loss = self.d_loss_real + self.d_loss_fake + self.lambdaCls * self.d_loss_cls + self.d_loss_gp
        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('Discriminator/')]
        train_D = tf.train.AdamOptimizer(learning_rate=self.learningRateD, beta1=0.5, beta2=0.999)
        self.train_D_loss = train_D.minimize(self.d_loss, var_list=self.d_params)
        # gvs = self.train_D.compute_gradients(self.d_loss, var_list=self.d_params)
        # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        # self.train_D_loss = self.train_D.apply_gradients(capped_gvs)

        #-----------------accuracy--------------------------------------------------------------
        YCls_real_sigmoid = tf.sigmoid(YCls_real)
        predicted = tf.to_int32(YCls_real_sigmoid > 0.5)
        labels = tf.to_int32(self.realLabels)
        correct = tf.to_float(tf.equal(predicted, labels))
        hundred = tf.constant(100.0)
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), axis=0) * hundred
        #--------------------------------------------------------------------------------------


        #CLIP D WEIGHTS
        # theta_D is list of D's params
        self.clip_D = [p.assign(tf.clip_by_value(p, -self.clipD, self.clipD)) for p in self.d_params]


        # -----------------------------------------------------------------------------------------
        # ----------------------------Create G training pipeline-----------------------------------
        # -----------------------------------------------------------------------------------------
        #original to target and target to original domain
        #self.fakeX = self.Gen.recForward(self.realX, self.fakeLabelsOneHot)
        rec_x = self.Gen.recForward(self.fakeX,self.realLabelsOneHot)

        # compute losses
        #out_src, out_cls = self.Dis.forward(self.fakeX)
        self.g_loss_adv = - tf.reduce_mean(YSrc_fake)
        self.g_loss_cls = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.fakeLabels,logits=tf.squeeze(YCls_fake))) / self.batchSize

        self.g_loss_rec = tf.reduce_mean(tf.abs(self.realX - rec_x))
        # total G loss and optimize
        self.g_loss = self.g_loss_adv + self.lambdaCls * self.g_loss_cls + self.lambdaRec * self.g_loss_rec
        train_G = tf.train.AdamOptimizer(learning_rate=self.learningRateG, beta1=0.5, beta2=0.999)
        self.train_G_loss = train_G.minimize(self.g_loss)
        # gvs = self.train_G.compute_gradients(self.g_loss)
        # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        # self.train_G_loss = self.train_G.apply_gradients(capped_gvs)

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
        gloss = 0
        g_loss_adv = 0
        g_loss_cls = 0
        g_loss_rec = 0
        # test image
        img_test = normalize(sci.imread(os.path.join(self.DBpath, "002000_[1, 0, 0, 1, 0].jpg")))
        labels = np.array([0, 1, 0, 1, 0])
        img_test = np.stack([img_test])
        #----------------------------------------------
        # -----------------------------------------------------------------------------------------
        # -----------------------------------BEGIN TRAINING----------------------------------------
        # -----------------------------------------------------------------------------------------
        for e in range(self.epochs):
            index = 0
            for filename in (os.listdir(self.DBpath)):
                img = sci.imread(os.path.join(self.DBpath, filename))
                splits = filename.split('_')
                trueLabels.append(literal_eval(splits[1].split('.')[0]))

                # Normalization and random flip (data augmentation)
                img = normalize(img)
                if random.random() > 0.5:
                    img = np.fliplr(img)

                images.append(img)
                #print(filename, np.mean(img))
                if len(images) % self.batchSize == 0:

                    # Create fake labels and associated images
                    randomLabels = np.random.randint(2, size=(self.batchSize, self.numClass))
                    # realX_fakeLabels has to contain the original image with the labels to generate concatenated
                    imagesWithFakeLabels = stackLabels(images, randomLabels)
                    epsilon = np.random.rand()
                    # -----------------------------------------------------------------------------------------
                    # -----------------------------------TRAIN DISCRIMINATOR-----------------------------------
                    # -----------------------------------------------------------------------------------------
                    print("training discriminator...")

                    alpha = np.random.uniform(low=0, high=1.0)
                    _ = self.sess.run([self.train_D_loss, self.clip_D],
                                            feed_dict={
                                                       self.realLabels: trueLabels,
                                                       self.realX: np.stack(images),
                                                       self.fakeLabels: randomLabels,
                                                       self.fakeLabelsOneHot: stackLabelsOnly(randomLabels),
                                                       self.alphagp: alpha
                                                       })

                    # GRADIENT PENALTY
                    # _ = self.sess.run([self.train_D_gp,],
                    #                         feed_dict={
                    #                                    self.realLabels: trueLabels,
                    #                                    self.realX: np.stack(images),
                    #                                    self.fakeLabels: randomLabels,
                    #                                    self.fakeLabelsOneHot: stackLabelsOnly(randomLabels),
                    #                                    self.alphagp: alpha
                    #                                    })


                    if (index+1) % self.g_skip_step == 0:
                        # -----------------------------------------------------------------------------------------
                        # -----------------------------------TRAIN GENERATOR---------------------------------------
                        # -----------------------------------------------------------------------------------------
                        print("training generator...")
                        _, = self.sess.run([self.train_G_loss],
                                                feed_dict={
                                                           self.realLabelsOneHot: stackLabelsOnly(trueLabels),
                                                           self.realX: np.stack(images),
                                                           self.fakeLabels: randomLabels,
                                                           self.fakeLabelsOneHot: stackLabelsOnly(randomLabels)})


                    # -----------------------------------------------------------------------------------------
                    # -----------------------------------PRINTING LOSSES---------------------------------------
                    # -----------------------------------------------------------------------------------------
                    print("printing losses...")
                    dloss, d_loss_real, d_loss_fake, d_loss_gp, d_loss_cls, accuracy, gloss, g_loss_adv, g_loss_cls, g_loss_rec = self.sess.run(
                                            [
                                             self.d_loss,
                                             self.d_loss_real,
                                             self.d_loss_fake,
                                             self.d_loss_gp,
                                             self.d_loss_cls,
                                             self.accuracy,
                                             self.g_loss,
                                             self.g_loss_adv,
                                             self.g_loss_cls,
                                             self.g_loss_rec],
                                            feed_dict={
                                                       self.realLabels: trueLabels,
                                                       self.realLabelsOneHot: stackLabelsOnly(trueLabels),
                                                       self.realX: np.stack(images),
                                                       self.fakeLabels: randomLabels,
                                                       self.fakeLabelsOneHot: stackLabelsOnly(randomLabels),
                                                       self.alphagp: alpha
                                                       })

                    images = []
                    trueLabels = []

                    #print("Loss = " , dloss + gloss, " ", "Dloss = " , dloss, " ", "Gloss = ", gloss, "Epoch =", e)
                    #print("Dloss = " , dloss, " d_loss_real: ", d_loss_real, " d_loss_fake: ", d_loss_fake, " gradient penalty: ", _gradient_penalty, " d_loss_cls: ", d_loss_cls)
                    #print("YCls_real: ")
                    #print(YCls_real)
                    #print([np.mean(i) for i in gradLoss])
                    print("---------------------------")
                    print("Picture: ", index, "Accuracy: ",accuracy, "Dloss = " , dloss, " d_loss_real: ", d_loss_real, " d_loss_fake: ", d_loss_fake, " d_loss_gp: ", d_loss_gp, " d_loss_cls: ", d_loss_cls)
                    print("Picture: ", index, "Accuracy: ",accuracy, "Gloss = ", gloss, "g_loss_Adv: ", g_loss_adv, "g_loss_cls: ", g_loss_cls, "g_loss_rec: ", g_loss_rec)

                    #print("Picture: ", i, "Accuracy: ",accuracy, "Loss = " , dloss + gloss, " ", "Dloss = " , dloss, " ", "Gloss = ", gloss, "Epoch =", e)

                    print("---------------------------")
                    index+=1
                #save images

                if (index % 50 == 0):
                    generatedImage = np.squeeze(self.sess.run([self.fakeX], feed_dict={self.realX: img_test,self.fakeLabelsOneHot: stackLabelsOnly([labels])}), axis=0)

                    sci.imsave('D:/GANSProject/samples/outfile' + str(index) + '.jpg', denormalize(generatedImage))


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

            #save samples every 500 images




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

            # Get tensors
            recov_fakeX = sess.graph.get_tensor_by_name("fakeX:0")
            recov_realX = sess.graph.get_tensor_by_name("realX:0")
            recov_fakeLabelsOneHot = sess.graph.get_tensor_by_name("fakeLabelsOneHot:0")

            img = np.reshape(img, (1, 128, 128, 3))
            generatedImage = np.squeeze(sess.run([recov_fakeX], feed_dict={recov_realX: np.stack(img),
                                                                                recov_fakeLabelsOneHot: stackLabelsOnly(
                                                                                   labels)}), axis=0)

            # sci.imsave('img.jpg', img)
            # img = normalize(img)
            # sci.imsave('out.jpg', denormalize(img))

            sci.imsave('outfile1.jpg', denormalize(generatedImage))
