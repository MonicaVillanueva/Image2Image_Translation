from pipeline import Pipeline
import numpy as np
if __name__ == '__main__':

    p = Pipeline()
    p.init_model()
    # p.train()
    lab = np.array([1,0,0,0,1])
    p.test(labels=np.reshape(lab, (1,5)))




