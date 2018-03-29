import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
from tensorflow.python.ops.gradients_impl import _hessian_vector_product
# from negd import NEGD

class Model():
    def __init__(self, flags, is_NEGD=True):
        self.FLAGS = flags
        self.alpha = self.FLAGS.alpha
        learning_rate = self.FLAGS.eta

        # Placeholder
        self.X = tf.placeholder(tf.float32, shape=[None, self.FLAGS.input_dim], name='X')
        self.Z = tf.placeholder(tf.float32, shape=[None, self.FLAGS.z_dim], name='Z')

        self.g_z, self.theta_g, self.theta_d, self.D_loss, self.G_loss = self.training_procedure()

        ## Now we choose the loss function that we want to use:
        same = True
        if same:
            # Same function for Generator and discriminator
            gen_loss = -self.D_loss
            discr_loss = self.D_loss
        else:
            # Non-saturating GAN: different loss function for the Generator
            gen_loss = self.G_loss
            discr_loss = self.D_loss

        # optimizer_gd = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optimizer_gd = tf.train.AdagradOptimizer(learning_rate=learning_rate*10)
        opt_2 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.g_solver, self.g_gradients = self.minimize(optimizer_gd, gen_loss, self.theta_g)
        self.d_solver, self.d_gradients = self.minimize(opt_2, discr_loss, self.theta_d)

        self.v_gen_min, self.e_gen_min = self.get_min_eigvec(gen_loss, self.theta_g)
        self.v_discr_max, self.e_discr_max = self.get_min_eigvec(discr_loss, self.theta_d)

        self.nc_step_gen = self.nc_step(self.v_gen_min, self.e_gen_min, self.theta_g, self.FLAGS.alpha*10)
        self.nc_step_discr = self.nc_step(self.v_discr_max, self.e_discr_max, self.theta_d, self.FLAGS.alpha)

        self.batch_gradient = tf.sqrt(tf.square(self.g_gradients) + tf.square(self.d_gradients))

        self.tb_gd, self.tb_nc, self.tb_test = self.add_tboard()


    ## Graph construction

    def training_procedure(self):
        # Training Procedure
        g_z, theta_g = self.generator(self.Z)
        d_logit_real, theta_d = self.discriminator(self.X)
        d_logit_synth, _ = self.discriminator(g_z, reuse=True)

        # Loss for Discriminator: - log D(x) - log(1-D(G(z)))
        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_synth, labels=tf.zeros_like(d_logit_synth)))
        D_loss = D_loss_real + D_loss_fake

        # Loss for Generator: -log(D(G(z)))
        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_synth, labels=tf.ones_like(d_logit_synth)))

        return g_z, theta_g, theta_d, D_loss, G_loss

    def generator(self, z, reuse=False):
        scope = "generator"
        with tf.variable_scope(scope):
            g_hidden = tf.layers.dense(z, self.FLAGS.G_h1, activation=self.leaky_relu, kernel_initializer=xavier_initializer(), name="G1", reuse=reuse)
            if self.FLAGS.G_h2 is not None:
                g_hidden = tf.layers.dense(g_hidden, self.FLAGS.G_h2, activation=self.leaky_relu, kernel_initializer=xavier_initializer(), name="Gh2", reuse=reuse)
            g_z = tf.layers.dense(g_hidden, self.FLAGS.input_dim, kernel_initializer=xavier_initializer(), name="G2", reuse=reuse)
            if self.FLAGS.dataset == "mnist":
                g_z = tf.nn.sigmoid(g_z, name="g_z")
            else:
                g_z = tf.identity(g_z, name="g_z")
        theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        return g_z, theta_g

    def discriminator(self, x, reuse=False):
        scope = "discriminator"
        with tf.variable_scope(scope):
            d_hidden = tf.layers.dense(x, self.FLAGS.D_h1, activation=self.leaky_relu, kernel_initializer=xavier_initializer(), reuse=reuse, name="D1")
            if self.FLAGS.D_h2 is not None:
                d_hidden = tf.layers.dense(d_hidden, self.FLAGS.D_h2, activation=self.leaky_relu,
                                           kernel_initializer=xavier_initializer(), name="Dh2", reuse=reuse)
            d_logit = tf.layers.dense(d_hidden, 1, kernel_initializer=xavier_initializer(), reuse=reuse, name="D2")
        theta_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        return d_logit, theta_d

    def leaky_relu(self, x, alpha=0.2):
        return tf.maximum(tf.minimum(0.0, alpha * x), x)

    ## Optimization

    # For GD Step
    def minimize(self, optimizer, loss, var_list):
        grads_vars = optimizer.compute_gradients(loss, var_list=var_list)
        list_of_gradmat = tf.concat([tf.reshape(gv[0], [-1]) for gv in grads_vars], 0)
        squared_gradients = tf.reduce_sum(tf.square(list_of_gradmat))
        solver = optimizer.apply_gradients(grads_vars)
        return solver, squared_gradients

    # For Negative Curvature Step
    def nc_step(self, v, e, _vars, lr):
        d = lr* np.abs(e) * v
        d_zero = tf.zeros_like(d)
        # If the minimum eigenvalue is < 0: then we do the update; otherwise just update with a zero vector
        assign_ops = tf.cond(e < 0, lambda: self._apply_step(d, _vars), lambda: self._apply_step(d_zero, _vars))
        return assign_ops

    def get_min_eigvec(self, loss, vars):
        iterations = 10
        eps = 3
        v = [self._get_initial_vector(vars)]
        eigvals = []
        grad = self._list_to_tensor(tf.gradients(loss, vars))
        for i in range(iterations):
            # Power iteration with the shifted Hessian
            v_new = self._list_to_tensor(_hessian_vector_product(loss, vars, self._tensor_to_list(v[i], vars)))
            v.append(eps*v[i] - v_new)
            v[i+1] = self._normalize(v[i+1])

            # Get corresponding eigenvalue
            eigval = tf.reduce_sum(tf.multiply(v[i], self._list_to_tensor(
                _hessian_vector_product(loss, vars, self._tensor_to_list(v[i], vars)))))
            eigvals.append(eigval)

        idx = iterations -1 #tf.cast(tf.argmin(eigvals[3:iterations-1]), tf.int32)
        e = tf.gather(eigvals, idx)
        v = tf.gather(v, idx)

        _sign = -tf.sign(tf.reduce_sum(tf.multiply(grad, v)))
        v *= _sign

        return v, e

    def _get_initial_vector(self, _vars):
        v = tf.random_uniform([self.get_num_weights(_vars)])
        v = self._normalize(v)
        return v

    def _normalize(self, v):
        v /= tf.sqrt(tf.reduce_sum(tf.square(v), 0, keep_dims=True))
        return v

    def _tensor_to_list(self, tensor, _vars):
        shape = [layer.get_shape().as_list() for layer in _vars]
        tensor_list = []
        offset = 0
        for sh in shape:
            end = offset + np.prod(sh)
            tensor_list.append(tf.reshape(tensor[offset:end], sh))
            offset = end
        return tensor_list

    def _list_to_tensor(self, _list):
        _tensor = tf.concat([tf.reshape(layer, [-1]) for layer in _list], axis=0)
        return _tensor

    ## Tensorboard for Visualization

    def add_tboard(self):
        # things to merge in the gd step
        tf.summary.scalar('Discr. Loss', self.D_loss, collections=["gd_step"])
        tf.summary.scalar("Gen. Loss", self.G_loss, collections=["gd_step"])
        tf.summary.scalar("Gen. Gradient", self.g_gradients, collections=["gd_step"])
        tf.summary.scalar("Discr. Gradient", self.d_gradients, collections=["gd_step"])

        # things to merge in the negative curvature step
        tf.summary.scalar('Discr. max Eigenvalue', -self.e_discr_max, collections=["nc_step"])
        tf.summary.scalar("Gen. min Eigenvalue", self.e_gen_min, collections=["nc_step"])

        # things to merge in the test step
        tf.summary.scalar('Batch Gradient', self.batch_gradient, collections=["test"])
        gen_image = tf.reshape(self.g_z[:5, :], [5, 28, 28, 1])
        tf.summary.image("Generated image", gen_image, collections=["test"])

        summary_gd = tf.summary.merge_all('gd_step')
        summary_nc = tf.summary.merge_all('nc_step')
        summary_test = tf.summary.merge_all('test')

        return summary_gd, summary_nc, summary_test

    ## Helper Functions

    def _apply_step(self, d, _vars):
        # Reshape d to fit the layer shapes
        shape = [layer.get_shape().as_list() for layer in _vars]
        d_list = []
        offset = 0
        for sh in shape:
            end = offset + np.prod(sh)
            d_list.append(tf.reshape(d[offset:end], sh))
            offset = end
        # Do the assign operations
        assign_ops = []
        for i in range(len(_vars)):
            assign_ops.append(tf.assign_add(_vars[i], d_list[i]))
        return assign_ops

    def get_num_weights(self, params):
        shape = [layer.get_shape().as_list() for layer in params]
        n = 0
        for sh in shape:
            n_layer = 1
            for dim in sh:
                n_layer *= dim
            n += n_layer
        return n
