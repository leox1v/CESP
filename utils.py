import tensorflow as tf
import numpy as np
import os
import pickle


def sample_Z(m, n):
    '''Gaussian prior for G(Z)'''
    return np.random.normal(size=[m,n])

def setup_directories(*dirs):
    for _dir in dirs:
        if not os.path.exists(_dir):
            os.makedirs(_dir)

def get_flags(dataset, exp_no):
    path = "results/{}/exp_{}/checkpoint/flags".format(dataset, exp_no)
    with open(path, 'rb') as file:
        FLAGS = pickle.load(file)
    return FLAGS

def set_seed(seed=None):
    if seed is None:
        seed = np.random.randint(0, 100)
    print("Seed: {}".format(seed))
    np.random.seed(seed)
    tf.set_random_seed(seed)

class Helper():
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.exp_dir = ""
        self.exp_no = 0

    def setup_directories(self):
        result_dir = "results/" + self.FLAGS.dataset + "/"
        setup_directories(result_dir)

        i = 0
        self.exp_dir = result_dir + "exp_{}/".format(i)
        while os.path.exists(self.exp_dir):
            i += 1
            self.exp_dir = result_dir + "exp_{}/".format(i)
        self.exp_no = i
        os.makedirs(self.exp_dir)

        self.FLAGS.summaries_dir = self.exp_dir + self.FLAGS.summaries_dir

        setup_directories(self.FLAGS.summaries_dir)

        # with open(self.exp_dir + "_info.txt", 'w') as f:
        #     f.write(str(self.FLAGS.__flags))

        # with open(self.FLAGS.checkpoint_dir + "flags", 'wb') as file:
        #     pickle.dump(self.FLAGS, file, protocol=pickle.HIGHEST_PROTOCOL)

        return self.FLAGS



