import tensorflow as tf
from utils.utils import xavier_init
from models.brs_cpacgan.ops import *

class Generator:
    def __init__(self, Z_dim, data_dim, pac_num, mode):
        self.pac_num = pac_num
        self.mode = mode
        
        self.G_W1 = tf.Variable(xavier_init([Z_dim, 128]), name="W1")
        self.G_b1 = tf.Variable(tf.zeros(shape=[128]), name="b1")

        self.G_W2 = tf.Variable(xavier_init([128, data_dim]), name="W2")
        self.G_b2 = tf.Variable(tf.zeros(shape=[data_dim]), name="b2")

        self.Z = []
        for i in range(self.pac_num):
            self.Z.append(tf.placeholder(tf.float32, shape=[None, Z_dim]))

        self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]
        self.size = [self.theta_G[t].get_shape().as_list() for t in range(len(self.theta_G))]
        self.build()
    
    def build(self):
        self.G_sample = []
        for i in range(self.pac_num):
            G_h1 = tf.nn.relu(tf.matmul(self.Z[i], self.G_W1) + self.G_b1)
            G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
            G_prob = tf.nn.sigmoid(G_log_prob)
            if self.mode == "smooth" or self.mode == "gradient":
                self.G_sample.append(G_prob)
            elif self.mode == "binary":
                self.G_sample.append(tf.to_float(G_prob > 0.5))
            elif self.mode == "multilevel":
                G_sample_tmp = tf.to_int32(G_prob > 1/ 10.0)
                for i in range(2, 10):
                    G_sample_tmp = G_sample_tmp + tf.to_int32(G_prob > i/ 10.0)
                self.G_sample.append(tf.to_float(G_sample_tmp) / tf.constant(10.0))
            else:
                print("Incompatiable mode!")
                exit()

    def update(self):
        return

class DCGenerator:
    def __init__(self, Z_dim, data_dim, pac_num, mode, batch_size):
        self.pac_num = pac_num
        self.mode = mode
        self.batch_size = batch_size

        self.depths=[1024, 512, 256, 128, 1]
        self.s_size = 2

        self.Z = []
        for i in range(self.pac_num):
            self.Z.append(tf.placeholder(tf.float32, shape=[None, Z_dim]))
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.theta_G = []
        self.size = []
        self.reuse = False
        self.gf_dim = 64
        self.build()
    
    def build(self):
        def conv_out_size_same(size, stride):
            return int(math.ceil(float(size) / float(stride)))

        self.G_sample = []
        
        with tf.variable_scope('g', reuse=tf.AUTO_REUSE):
            for i in range(self.pac_num):
                s_h, s_w = 28, 28
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
                
                # project `z` and reshape
                z, _, _ = linear(
                    self.Z[i], self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

                h0 = tf.reshape(
                    z, [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = lrelu(self.g_bn0(h0))

                h1, _, _ = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
                h1 = lrelu(self.g_bn1(h1))

                h2, _, _ = deconv2d(
                    h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
                h2 = lrelu(self.g_bn2(h2))

                h3, _, _ = deconv2d(
                    h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
                h3 = lrelu(self.g_bn3(h3))

                h4, _, _ = deconv2d(
                    h3, [self.batch_size, s_h, s_w, 1], name='g_h4', with_w=True)

                G_prob = tf.nn.sigmoid(h4)
                
                if self.mode == "smooth" or self.mode == "gradient":
                    self.G_sample.append(G_prob)
                elif self.mode == "binary":
                    self.G_sample.append(tf.to_float(G_prob > 0.5))
                elif self.mode == "multilevel":
                    G_sample_tmp = tf.to_int32(G_prob > 1/ 10.0)
                    for i in range(2, 10):
                        G_sample_tmp = G_sample_tmp + tf.to_int32(G_prob > i/ 10.0)
                    self.G_sample.append(tf.to_float(G_sample_tmp) / tf.constant(10.0))
                else:
                    print("Incompatiable mode!")
                    exit()

        self.theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        self.size = [self.theta_G[t].get_shape().as_list() for t in range(len(self.theta_G))]

    def update(self):
        return
