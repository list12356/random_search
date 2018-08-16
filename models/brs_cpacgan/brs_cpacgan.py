import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import matplotlib
import csv
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

from scipy import stats
from time import sleep

from utils.utils import sample_Z
from utils.utils import plot_mnist
from utils.utils import SharedNoiseTable
from models.brs_cpacgan.generator import Generator
from models.brs_cpacgan.generator import DCGenerator
from models.brs_cpacgan.discriminator import Discriminator
from models.brs_cpacgan.discriminator import DCDiscriminator
from models.brs_cpacgan.dataloader import MNISTDataLoader
from models.brs_cpacgan.dataloader import SyntheticDataLoader
from models.brs_cpacgan.dataloader import PatternDataLoader

import multiprocessing as mp

        
class BRSCPacGAN:
    def __init__(self, out_dir = "out_brs_cpac", alpha=0.2, v=0.02, _lambda = 1.0, 
                sigma = 0, mode = "binary", pac_num = 1, gan_structure="vanila",
                dataset="mnist", wasserstein=0):
        self.out_dir = out_dir
        self.save_step = 1000
        self.Z_dim = 100
        self.search_num = 64
        self.num_workers = 8
        self.alpha = alpha
        self.v = v
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

        with tf.variable_scope("generator"):
            if gan_structure == "vanila":
                self.G = Generator(self.Z_dim,self.data_dim, self.pac_num, self.mode)
            elif gan_structure == "dc":
                self.G = DCGenerator(self.Z_dim,self.data_dim, self.pac_num, self.mode, self.batch_size)
        self.G_sample = self.G.G_sample

        self.S_sample = [None] * self.num_workers
        self.S  = [None] * self.num_workers
        for w in range(self.num_workers):
            with tf.variable_scope("searcher_" + str(w)):
                if gan_structure == "vanila":
                    self.S[w] = Generator(self.Z_dim,self.data_dim, self.pac_num, self.mode)
                elif gan_structure == "dc":
                    self.S[w] = DCGenerator(self.Z_dim,self.data_dim, self.pac_num, self.mode, self.batch_size)
            self.S_sample[w] = self.S[w].G_sample

        self.X = []
        for p in range(self.pac_num):
            self.X.append(tf.placeholder(tf.float32, shape=[self.batch_size, self.data_dim]))

        with tf.variable_scope("discriminator"):
            if gan_structure == "vanila":
                self.D = Discriminator(self.pac_num, self.data_dim)
            elif gan_structure == "dc":
                self.D = DCDiscriminator(self.pac_num, self.data_dim, self.batch_size)
        
        D_real, D_logit_real = self.D.build(self.X)
        D_fake, D_logit_fake = self.D.build(self.G_sample)

        if self.wasserstein == 0:
            self.D_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(D_real, 1e-16, 1.0))\
                 + tf.log(tf.clip_by_value(1. - D_fake, 1e-16, 1.)))
            self.G_loss = tf.reduce_mean(tf.log(tf.clip_by_value(D_fake, 1e-16, 1.0)))
        else:
            self.D_loss = -tf.reduce_mean(D_logit_real) + tf.reduce_mean(D_logit_fake)
            self.G_loss = tf.reduce_mean(D_logit_fake)
            
        self.S_loss = [None] * self.num_workers
        for w in range(self.num_workers):
            S_fake, S_logit_fake = self.D.build(self.S_sample[w])
            if self.wasserstein == 0:
                loss = tf.reduce_mean(tf.log(S_fake), name="s_loss_" + str(w))
            else:
                loss = tf.reduce_mean(S_logit_fake, name="s_loss_" + str(w))
            self.S_loss[w] = loss
        
        
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
        self.clip_D = tf.group(*self.clip_D)

        self.train_num = 1000000
        if self.mode == "gradient":
            if self.wasserstein == 1:
                self.G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
                                .minimize(-self.G_loss, var_list=self.G.theta_G))
            else:
                self.G_solver = tf.train.AdamOptimizer().minimize(-self.G_loss, var_list=self.G.theta_G)

        else:
            # self.update_Sp = [None] * self.num_workers
            # self.update_Sn = [None] * self.num_workers
            self.update_Sp = []
            self.update_Sn = []
            self.delta_ph = [None] * self.num_workers
            self.update_G = [None] * len(self.G.theta_G)
            self.reward_ph = tf.placeholder(tf.float32, shape=(self.num_workers))
            self.delta = [None] * len(self.G.theta_G)
            print('creating')
            for w in range(self.num_workers):
                # self.update_Sp[w] = []
                # self.update_Sn[w] = []
                self.delta_ph[w] = []
                for t in range(len(self.G.theta_G)):
                    with tf.device("/device:GPU:0"):
                        delta_ph = tf.placeholder(tf.float32, shape=self.G.size[t])
                        self.delta_ph[w].append(delta_ph)
                        self.update_Sp.append(tf.assign(self.S[w].theta_G[t], self.G.theta_G[t] + delta_ph, validate_shape=False))
                        self.update_Sn.append(tf.assign(self.S[w].theta_G[t], self.G.theta_G[t] - delta_ph, validate_shape=False))

                        # perturb_shape = self.G.size[t]
                        # indices = list(range(w, perturb_shape[0], self.num_workers))
                        # perturb_shape[0] = len(indices)
                        # delta_ph = tf.placeholder(tf.float32, shape=perturb_shape)
                        # self.delta_ph[w].append(delta_ph)

                        # self.update_Sp[w].append(tf.scatter_add(self.S[w].theta_G[t], indices, delta_ph))
                        # self.update_Sn[w].append(tf.scatter_sub(self.S[w].theta_G[t], indices, delta_ph))

            self.delta_ph = np.array(self.delta_ph)
            for t in range(len(self.G.theta_G)):
                dims = len(self.G.size[t])
                # TODO: add sigma_R
                update = tf.reduce_sum(tf.reshape(self.reward_ph, [self.num_workers] + [1] * dims) * \
                    tf.stack(self.delta_ph[:,t].tolist()), axis=0) * self.alpha / (self.num_workers * self.v)
                self.update_G[t] = tf.assign(self.G.theta_G[t], self.G.theta_G[t] + update)
            
            self.S_reward = tf.stack(self.S_loss, 0)
            with tf.device("/device:GPU:0"):
                self.update_Sn_op = tf.group(*self.update_Sn)
                self.update_Sp_op = tf.group(*self.update_Sp)
                self.update_G_op = tf.group(*self.update_G)
            print('finished')

        print("delta")
        self.noise = SharedNoiseTable()
        self.rs = np.random.RandomState()
        print("end")
        
        if os.path.exists(self.out_dir) == False:
            os.makedirs(self.out_dir)
        # workers = []
        # for w in range(self.num_workers):
        #     workers.append("localhost:" + str(2223+w))
        # cluster = tf.train.ClusterSpec({"worker": workers})
        # self.sess = []
        # for w in range(self.num_workers):
        #     server = tf.train.Server(cluster,
        #                      job_name="worker",
        #                      task_index=w)
        #     self.sess.append(tf.Session(target=server.target))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 44
        config.inter_op_parallelism_threads = 44
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        img_num = 0

        sigma_R = 1.
        saver = tf.train.Saver()

        log_file = open(self.out_dir+"/exp-log.csv", "w+")
        log_fileds = ['epoch', 'D_loss', 'G_loss', 'sigma']
        writer = csv.DictWriter(log_file, fieldnames=log_fileds)

        # manager = mp.Manager()
        # delta = manager.list(self.delta)

        for it in range(self.train_num):
            if it % self.save_step == 0:
                samples = self.sess.run(self.G_sample[0], feed_dict={self.G.Z[0]: sample_Z(self.batch_size, self.Z_dim)})
                samples = samples[:16]
                fig = plot_mnist(samples)
                plt.savefig(self.out_dir + '/{}.png'.format(str(img_num).zfill(5)), bbox_inches='tight')
                plt.close(fig)

                X_t, _ = self.dataloader.next_batch()
                X_t = X_t[:16]
                fig = plot_mnist(X_t)
                plt.savefig(self.out_dir + '/{}_true.png'.format(str(img_num).zfill(5)), bbox_inches='tight')
                plt.close(fig)
                
                # saver.save(self.sess, self.out_dir + '/{}_model.ckpt'.format(str(img_num).zfill(5)))
                img_num += 1

            X_mb = []
            for p in range(self.pac_num):
                X_t, _ = self.dataloader.next_batch()
                if self.gan_structure == "dc":
                    X_t = np.reshape(X_t, [self.batch_size, 28, 28, 1])
                X_mb.append(X_t)

            feed_S = {}
            feed_G = {}
            for p in range(self.pac_num):
                for w in range(self.num_workers):
                    idx = self.noise.sample_index(self.rs, self.Z_dim * self.batch_size)
                    feed_S[self.S[w].Z[p]] = np.reshape(self.noise.get(idx, self.Z_dim * self.batch_size), 
                        [self.batch_size, self.Z_dim])
                feed_G[self.G.Z[p]] = np.reshape(self.noise.get(idx, self.Z_dim * self.batch_size), 
                        [self.batch_size, self.Z_dim])

            # update = manager.list([None] * len(self.G.theta_G))

            if self.mode == "gradient":
                _, G_loss_curr = self.sess.run([self.G_solver, self.G_loss], feed_dict=feed_G)
            else:
                delta = []
                for t in range(len(self.G.theta_G)):
                    dim = self.num_workers
                    for x in self.G.size[t]:
                        dim = dim * x
                    idx = self.noise.sample_index(self.rs, dim)
                    delta.append(self.v* np.reshape(self.noise.get(idx, dim),
                        [self.num_workers] + self.G.size[t]))
                # for w in range(self.num_workers):
                #     for t in range(len(self.G.theta_G)):
                #         self.sess.run(self.update_Sp[w][t], feed_dict={self.delta_ph[w][t]: delta[t][w]})

                feed_update = {}
                for w in range(self.num_workers):
                    for t in range(len(self.G.theta_G)):
                        feed_update[self.delta_ph[w][t]] = delta[t][w]
                
                self.sess.run(self.update_Sp_op, feed_dict=feed_update)
                reward = self.sess.run(self.S_reward, feed_dict=feed_S)
                # for w in range(self.num_workers):
                #     for t in range(len(self.G.theta_G)):
                #         self.sess.run(self.update_Sn[w][t], feed_dict={self.delta_ph[w][t]: delta[t][w]})
                
                self.sess.run(self.update_Sn_op, feed_dict=feed_update)
                tmp = self.sess.run(self.S_reward, feed_dict=feed_S)
                sigma = np.concatenate([reward, tmp])
                reward = reward - tmp

                feed_update[self.reward_ph] = reward
                self.sess.run(self.update_G_op, feed_dict=feed_update)
            
                if self.sigma == 1:
                    sigma_R = np.std(sigma)
                else:
                    sigma_R = 1.

            G_loss_curr = self.sess.run(self.G_loss, feed_dict=feed_G)
            
            for p in range(self.pac_num):
                feed_G[self.X[p]] = X_mb[p]
            _, D_loss_curr = self.sess.run([self.D_solver, self.D_loss], feed_dict=feed_G)

            if it % 100 == 0:
                print('Iter: {}'.format(it))
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