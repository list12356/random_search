import tensorflow as tf
from utils.utils import xavier_init
from models.brs_cpacgan.ops import *

class Discriminator:
    def __init__(self, pac_num, data_dim):
        self.D_W1 = tf.Variable(xavier_init([data_dim * pac_num, 128]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[128]))

        self.D_W2 = tf.Variable(xavier_init([128, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))

        self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

    def build(self, x):
        x = tf.concat(x, axis=-1)
        D_h1 = tf.nn.relu(tf.matmul(tf.to_float(x), self.D_W1) + self.D_b1)
        D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
        D_prob = tf.nn.sigmoid(D_logit)

        return D_prob, D_logit

class DCDiscriminator:
    def __init__(self, pac_num, data_dim, batch_size):
        self.depths = [1 * pac_num, 64, 128, 256, 512]
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.theta_D = []
        self.df_dim = 64
        self.reuse = False
        self.batch_size = batch_size

    def build(self, inputs):

        training = True
        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)
        
        for i in range(len(inputs)):
            inputs[i] = tf.reshape(inputs[i], [-1, 28, 28, 1]) 

        image = tf.concat(inputs, axis = -1)

        with tf.variable_scope('d', reuse=self.reuse):
            # convolution x 4
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

        self.reuse = True
        self.theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
        return tf.nn.sigmoid(h4), h4
