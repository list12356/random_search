import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import matplotlib
import csv
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from utils.utils import sample_Z
from utils.utils import plot_mnist
from models.brs_cpacgan.generator import Generator
from models.brs_cpacgan.generator import DCGenerator
from models.brs_cpacgan.discriminator import Discriminator
from models.brs_cpacgan.discriminator import DCDiscriminator
from models.brs_cpacgan.dataloader import MNISTDataLoader
from models.brs_cpacgan.dataloader import SyntheticDataLoader
from models.brs_cpacgan.dataloader import PatternDataLoader



class BRSCPacGAN:
    def __init__(self, out_dir = "out_brs_cpac", alpha=0.2, v=0.02, _lambda = 1.0, 
                sigma = 0, mode = "binary", pac_num = 1, gan_structure="vanila",
                dataset="mnist", wasserstein=0):
        self.out_dir = out_dir
        self.save_step = 100
        self.Z_dim = 10
        self.search_num = 64
        self.alpha = alpha
        self.v = 0.02
        self._lambda = _lambda
        self.mode = mode
        # self.mnist = input_data.read_data_sets('./data/MNIST_data', one_hot=True)
        self.dataset = dataset
        self.restore = False
        self.D_lr = 1e-4
        self.batch_size = 64
        if self.dataset == "mnist":
            self.dataloader = MNISTDataLoader(self.batch_size, self.mode)
            self.data_dim = 784
        elif self.dataset == "synthetic":
            self.dataloader = SyntheticDataLoader(self.batch_size)
            self.data_dim = 10
        elif self.dataset == "pattern":
            self.dataloader = PatternDataLoader(self.batch_size)
            self.data_dim = 784
        else:
            print("unknown dataset!")
            exit(0)
        self.pac_num = pac_num
        self.sigma = sigma
        self.gan_structure = gan_structure
        self.wasserstein = wasserstein

        if gan_structure == "vanila":
            self.G = Generator(self.Z_dim,self.data_dim, self.pac_num, self.mode)
        elif gan_structure == "dc":
            self.G = DCGenerator(self.Z_dim,self.data_dim, self.pac_num, self.mode, self.batch_size)
        self.G_sample = self.G.G_sample

        if gan_structure == "vanila":
            self.S = Generator(self.Z_dim,self.data_dim, self.pac_num, self.mode)
        elif gan_structure == "dc":
            self.S = DCGenerator(self.Z_dim,self.data_dim, self.pac_num, self.mode, self.batch_size)
        self.S_sample = self.S.G_sample

        self.X = []
        for p in range(self.pac_num):
            self.X.append(tf.placeholder(tf.float32, shape=[self.batch_size, self.data_dim]))

        if gan_structure == "vanila":
            self.D = Discriminator(self.pac_num, self.data_dim)
        elif gan_structure == "dc":
            self.D = DCDiscriminator(self.pac_num, self.data_dim, self.batch_size)
        D_real, D_logit_real = self.D.build(self.X)
        D_fake, D_logit_fake = self.D.build(self.G_sample)
        S_fake, S_logit_fake = self.D.build(self.S_sample)

        if self.wasserstein == 0:
            self.D_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(D_real, 1e-16, 1.0)) + tf.log(tf.clip_by_value(1. - D_fake, 1e-16, 1.)))
            self.G_loss = tf.reduce_mean(tf.log(tf.clip_by_value(D_fake, 1e-16, 1.0))) * tf.constant(_lambda)
            self.S_loss = tf.reduce_mean(tf.log(S_fake)) * tf.constant(_lambda)
        else:
            self.D_loss = -tf.reduce_mean(D_logit_real) + tf.reduce_mean(D_logit_fake)
            self.G_loss = tf.reduce_mean(D_logit_fake)
            self.S_loss = tf.reduce_mean(S_logit_fake)
        # Alternative losses:
        # -------------------
        # D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        # D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        # D_loss = D_loss_real + D_loss_fake
        # G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
        if self.wasserstein == 1:
            self.D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
                            .minimize(self.D_loss, var_list=self.D.theta_D))
        else:
            self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.D.theta_D)
        self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.D.theta_D]

        if self.mode == "gradient":
            if self.wasserstein == 1:
                self.G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
                                .minimize(-self.G_loss, var_list=self.G.theta_G))
            else:
                self.G_solver = tf.train.AdamOptimizer().minimize(-self.G_loss, var_list=self.G.theta_G)

        else:
            self.update_Sp = []
            self.update_Sn = []
            self.update_G = []
            self.update_Gph = []
            self.delta_ph = []
            for t in range(len(self.G.theta_G)):
                self.delta_ph.append(tf.placeholder(tf.float32, shape=self.G.theta_G[t].get_shape().as_list()))
                self.update_Gph.append(tf.placeholder(tf.float32, shape=self.G.theta_G[t].get_shape().as_list()))
                self.update_Sp.append(tf.assign(self.S.theta_G[t], self.G.theta_G[t] + self.delta_ph[t]))
                self.update_Sn.append(tf.assign(self.S.theta_G[t], self.G.theta_G[t] - self.delta_ph[t]))
                self.update_G.append(tf.assign(self.G.theta_G[t], self.G.theta_G[t] + self.update_Gph[t]))

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def train(self):
        img_num = 0
        sigma_R = 1.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        log_file = open(self.out_dir+"/exp-log.csv", "w+")
        log_fileds = ['epoch', 'D_loss', 'G_loss', 'sigma']
        writer = csv.DictWriter(log_file, fieldnames=log_fileds)

        for it in range(1000000):
            if it % self.save_step == 0:
                samples = sess.run(self.G_sample[0], feed_dict={self.G.Z[0]: sample_Z(self.batch_size, self.Z_dim)})
                samples = samples[:16]
                fig = plot_mnist(samples)
                plt.savefig(self.out_dir + '/{}.png'.format(str(img_num).zfill(5)), bbox_inches='tight')
                plt.close(fig)

                X_t, _ = self.dataloader.next_batch()
                X_t = X_t[:16]
                fig = plot_mnist(X_t)
                plt.savefig(self.out_dir + '/{}_true.png'.format(str(img_num).zfill(5)), bbox_inches='tight')
                plt.close(fig)
                
                saver.save(sess, self.out_dir + '/{}_model.ckpt'.format(str(img_num).zfill(5)))
                img_num += 1

            X_mb = []
            for p in range(self.pac_num):
                X_t, _ = self.dataloader.next_batch()
                if self.gan_structure == "dc":
                    X_t = np.reshape(X_t, [self.batch_size, 28, 28, 1])
                X_mb.append(X_t)

            sample = []
            for p in range(self.pac_num):
                sample.append(sample_Z(self.batch_size, self.Z_dim))
            feed_S = {}
            feed_G = {}
            for p in range(self.pac_num):
                feed_S[self.S.Z[p]] = sample[p]
                feed_G[self.G.Z[p]] = sample[p]
            update = []
            reward_list = []
            reward_list_2 = []
            delta_list = []

            if self.mode == "gradient":
                _, G_loss_curr = sess.run([self.G_solver, self.G_loss], feed_dict=feed_G)
            else:
                for m in range(self.search_num):
                    delta = []
                    for t in range(len(self.G.theta_G)):
                        delta.append(self.v*np.random.normal(loc=0.0, scale=1.,
                                                    size=self.G.theta_G[t].get_shape().as_list()))
                    for t in range(len(self.G.theta_G)):
                        sess.run(self.update_Sp[t], feed_dict={self.delta_ph[t]: delta[t]})
                    reward = sess.run(self.S_loss, feed_dict=feed_S)
                    reward_list_2.append(reward)
                    for t in range(len(self.G.theta_G)):
                        sess.run(self.update_Sn[t], feed_dict={self.delta_ph[t]: delta[t]})
                    tmp = sess.run(self.S_loss, feed_dict=feed_S)
                    reward_list_2.append(tmp)
                    reward = reward - tmp

                    reward_list.append(reward)
                    delta_list.append(delta)

                if self.sigma == 1:
                    sigma_R = np.std(reward_list_2)
                else:
                    sigma_R = 1.
                if sigma_R == 0:
                    sigma_R = 1.

                for m in range(self.search_num):
                    if m == 0:
                        for t in range(len(self.G.theta_G)):
                            update.append(reward_list[m] * delta_list[m][t])
                    else:
                        for t in range(len(self.G.theta_G)):
                            update[t] += reward_list[m] * delta_list[m][t]
                for t in range(len(self.G.theta_G)):
                    sess.run(self.update_G[t], feed_dict={self.update_Gph[t]: update[t] * self.alpha / (self.search_num * sigma_R * self.v)})

            G_loss_curr = sess.run(self.G_loss, feed_dict=feed_G)
            
            for p in range(self.pac_num):
                feed_G[self.X[p]] = X_mb[p]
            _, D_loss_curr, _ = sess.run([self.D_solver, self.D_loss, self.clip_D], feed_dict=feed_G)

            if it % 100 == 0:
                print('Iter: {}'.format(it))
                # for t in range(1):
                #     print(update[0] * self.alpha / self.search_num)
                #     print(sess.run(G.theta_G[0]))
                print('D loss: {:.4}'.format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print('Sigma_R: {:.4}'.format(sigma_R))
                writer.writerow({
                    'epoch': it,
                    'D_loss':D_loss_curr,
                    'G_loss':G_loss_curr,
                    'sigma': sigma_R
                })
                print()
