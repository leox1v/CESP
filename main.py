from model import Model
from utils import *
from utils import Helper
from tensorflow.examples.tutorials.mnist import input_data
import pprint

flags = tf.app.flags
flags.DEFINE_integer("max_iter", "50000", "Maximum of iterations to train. [25]")
flags.DEFINE_integer("batch_size", "1000", "The size of batch images [64]")

flags.DEFINE_string("dataset", "mnist", "The dataset that is used. [mnist]")

flags.DEFINE_integer("input_dim", 784, "The dimension of the input samples. [2]")
flags.DEFINE_integer("z_dim", 10, "The size of latent vector z.[256]")
flags.DEFINE_integer("D_h1", 100, "The hidden dimension of the first layer of the Discriminator. [10]")
flags.DEFINE_integer("G_h1", 100, "The hidden dimension of the first layer of the Generator. [10]")
flags.DEFINE_integer("D_h2", None, "The hidden dimension of the first layer of the Discriminator. [10]")
flags.DEFINE_integer("G_h2", None, "The hidden dimension of the first layer of the Generator. [10]")

flags.DEFINE_string("summaries_dir", "tensorboard/", "Directory to use for the summary.")

flags.DEFINE_float("eta", 0.005 , "Learning rate for GD.")
flags.DEFINE_float("alpha", 0.01 , "Learning rate for NC.")

pp = pprint.PrettyPrinter()



def main(_):
    FLAGS = flags.FLAGS
    pp.pprint(flags.FLAGS.__flags)
    print(tf.__version__)

    # load data
    data = input_data.read_data_sets("MNIST_data/", one_hot=True)
    helper = Helper(FLAGS)
    FLAGS = helper.setup_directories()

    set_seed(42)
    test_data = test_batch(data, 1000, FLAGS)
    train(FLAGS, data, test_data)


def train(FLAGS, data, test_data):
    # specify the network
    model = Model(flags=FLAGS)

    tboard_dir = "tensorboard_exp/CEGD/"

    # initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(tboard_dir, graph=sess.graph)

    for it in range(FLAGS.max_iter + 1):
        # Data batch
        X, _ = data.train.next_batch(FLAGS.batch_size)
        Z = sample_Z(FLAGS.batch_size, FLAGS.z_dim)

        # Run negative curvature step
        _, _, summary_nc = sess.run([model.nc_step_gen, model.nc_step_discr, model.tb_nc],
                 feed_dict={model.X: X, model.Z: Z})

        # summary_nc = sess.run(model.tb_nc, feed_dict={model.X: X, model.Z: Z})

        # Run GD step
        _, _, summary_gd = sess.run([model.g_solver, model.d_solver, model.tb_gd], feed_dict={model.X: X, model.Z: Z})

        if it % 10 == 0:
            writer.add_summary(summary_nc, it)
            writer.add_summary(summary_gd, it)

        if it % 100 == 0:
            (X, Z) = test_data
            summary_test = sess.run(model.tb_test, feed_dict={model.X: X, model.Z: Z})
            writer.add_summary(summary_test, it)

def test_batch(data, test_n, FLAGS):
    X, _ = data.train.next_batch(test_n)
    Z = sample_Z(test_n, FLAGS.z_dim)
    return (X, Z)


if __name__ == '__main__':
    tf.app.run()