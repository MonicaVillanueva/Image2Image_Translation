from pipeline import Pipeline
from utils import stackLabels
import scipy.misc as sci
import os
import numpy as np
import scipy.misc
if __name__ == '__main__':

    p = Pipeline()
    p.init_model()
    p.train()

    img = sci.imread(os.path.join("D:processed/testImage/", "000038_[0, 0, 0, 1, 1].jpg"))

    randomLabels = np.random.randint(2, size=(1, 5))
    testImage = stackLabels([img], randomLabels)

    generatedImage = np.squeeze(p.sess.run([p.fakeX],feed_dict={p.realX_fakeLabels: testImage}))
    scipy.misc.imsave('C:/Users/Ferraat/Desktop/graph/outfile.jpg', generatedImage)


