import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import matplotlib
import csv
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from time import sleep

from utils.utils import sample_Z
from utils.utils import plot_mnist
from models.brs_cpacgan.generator import Generator
from models.brs_cpacgan.generator import DCGenerator
from models.brs_cpacgan.discriminator import Discriminator
from models.brs_cpacgan.discriminator import DCDiscriminator
from models.brs_cpacgan.dataloader import MNISTDataLoader
from models.brs_cpacgan.dataloader import SyntheticDataLoader
from models.brs_cpacgan.dataloader import PatternDataLoader

import multiprocessing as mp

def search_run(cluster, worker_n, search_num, indices, alpha, v, S_loss_name):
    
    theta_G = tf.trainable_variables(scope="generator")
    theta_S = tf.trainable_variables(scope="searcher_" + str(worker_n))
    S_loss = tf.get_default_graph().get_tensor_by_name(S_loss_name)
    delta_ph = [None] * len(indices)
    update_Sp = [None] * len(indices)
    update_Sn = [None] * len(indices)
    update_Gph = [None] * len(indices)
    update_G = [None] * len(indices)

    for i, t in enumerate(indices):
        delta_ph[i] = tf.placeholder(tf.float32, shape=theta_G[t].get_shape().as_list())
        update_Sp[i] = tf.assign(theta_S[t], theta_S[t] + delta_ph[i])
        update_Sn[i] = tf.assign(theta_S[t], theta_G[t] - delta_ph)
        
        update_Gph[i] = tf.placeholder(tf.float32, shape=theta_G[t].get_shape().as_list())
        update_G[i] = tf.assign(theta_G[t], theta_G[t] + update_Gph[t])
    
    server = tf.train.Server(cluster,
                             job_name="worker",
                             task_index=worker_n)
    sess = tf.Session(target=server.target)
    
    print("Worker %d: waiting for cluster connection..." % worker_n)
    sess.run(tf.report_uninitialized_variables())
    print("Worker %d: cluster ready!" % worker_n)
    
    while sess.run(tf.report_uninitialized_variables()):
        print("Worker %d: waiting for variable initialization..." % worker_n)
        sleep(1.0)
    print("Worker %d: variables initialized" % worker_n)

    for it in range(100000):
        delta = [None] * len(indices)
        update = [None] * len(indices)
        feed_S = None
        for i, t in enumerate(indices):
            delta_ph[i] = tf.placeholder(tf.float32, shape=theta_G[t].get_shape().as_list())
            delta[i] = v*np.random.normal(loc=0.0, scale=1.,
                            size=[search_num] + theta_G[t].get_shape().as_list())
        for m in range(search_num):
            for i, t in enumerate(indices):
                sess.run(update_Sp[i], feed_dict={delta_ph[i]: delta[i][m]})
            reward = sess.run(S_loss, feed_dict=feed_S)
            for i, t in enumerate(indices):
                sess.run(update_Sn[i], feed_dict={delta_ph[i]: delta[i][m]})
            tmp = sess.run(S_loss, feed_dict=feed_S)
            reward = reward - tmp
            for i, t in enumerate(indices):
                if m == 0:
                    update[i] = reward * delta[i][m]
                else:
                    update[i] += reward * delta[i][m]
        
        for i, t in enumerate(indices):
            sess.run(update_G[i], feed_dict={update_Gph[i]: update[i] * alpha / (search_num * v)})
    return

class BRSCPacGAN:
    def __init__(self, out_dir = "out_brs_cpac", alpha=0.2, v=0.02, _lambda = 1.0, 
                sigma = 0, mode = "binary", pac_num = 1, gan_structure="vanila",
                dataset="mnist", wasserstein=0):
        self.out_dir = out_dir
        self.save_step = 100
        self.Z_dim = 10
        self.search_num = 64
        self.num_workers = 4
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
            self.S_loss[w] = loss.name
        
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

        self.train_num = 1000000
        if self.mode == "gradient":
            if self.wasserstein == 1:
                self.G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
                                .minimize(-self.G_loss, var_list=self.G.theta_G))
            else:
                self.G_solver = tf.train.AdamOptimizer().minimize(-self.G_loss, var_list=self.G.theta_G)

        else:
            self.slices = [None] * self.num_workers
            self.update_Sp = [None] * self.num_workers
            self.update_Sn = [None] * self.num_workers
            self.delta_ph = [None] * self.num_workers
            self.update_G = [None] * len(self.G.theta_G)
            self.update_Gph = [None] * len(self.G.theta_G)
            self.delta = [None] * len(self.G.theta_G)
            for w in range(self.num_workers):
                self.slices[w] = []
                self.update_Sp[w] = []
                self.update_Sn[w] = []
                self.delta_ph[w] = []
            print('creating')
            for t in range(len(self.G.theta_G)):
                w = t % self.num_workers
                self.slices[w].append(t)
                delta_ph = tf.placeholder(tf.float32, shape=self.G.theta_G[t].get_shape().as_list())
                self.delta_ph[w].append(delta_ph)
                self.update_Sp[w].append(tf.assign(self.S[w].theta_G[t], self.G.theta_G[t] + delta_ph))
                self.update_Sn[w].append(tf.assign(self.S[w].theta_G[t], self.G.theta_G[t] - delta_ph))
                
                self.update_Gph[t] = tf.placeholder(tf.float32, shape=self.G.theta_G[t].get_shape().as_list())
                self.update_G[t] = tf.assign(self.G.theta_G[t], self.G.theta_G[t] + self.update_Gph[t])
                # self.delta[t] = self.v*np.random.normal(loc=0.0, scale=1.,\
                #         size=[self.train_num * self.search_num] + self.G.theta_G[t].get_shape().as_list())
            print('finished')
        
        if os.path.exists(self.out_dir) == False:
            os.makedirs(self.out_dir)
        workers = []
        for w in range(self.num_workers):
            workers.append("localhost:" + str(2223+w))
        cluster = tf.train.ClusterSpec({"worker": workers})
        self.sess = []
        for w in range(self.num_workers):
            server = tf.train.Server(cluster,
                             job_name="worker",
                             task_index=w)
            self.sess.append(tf.Session(target=server.target))

    def train(self):
        img_num = 0
        for w in range(self.num_workers):
            self.sess[w].run(tf.global_variables_initializer())

        sigma_R = 1.
        saver = tf.train.Saver()

        log_file = open(self.out_dir+"/exp-log.csv", "w+")
        log_fileds = ['epoch', 'D_loss', 'G_loss', 'sigma']
        writer = csv.DictWriter(log_file, fieldnames=log_fileds)

        # manager = mp.Manager()
        # delta = manager.list(self.delta)
        for it in range(self.train_num):
            if it % self.save_step == 0:
                samples = self.sess[0].run(self.G_sample[0], feed_dict={self.G.Z[0]: sample_Z(self.batch_size, self.Z_dim)})
                samples = samples[:16]
                fig = plot_mnist(samples)
                plt.savefig(self.out_dir + '/{}.png'.format(str(img_num).zfill(5)), bbox_inches='tight')
                plt.close(fig)

                X_t, _ = self.dataloader.next_batch()
                X_t = X_t[:16]
                fig = plot_mnist(X_t)
                plt.savefig(self.out_dir + '/{}_true.png'.format(str(img_num).zfill(5)), bbox_inches='tight')
                plt.close(fig)
                
                saver.save(self.sess[0], self.out_dir + '/{}_model.ckpt'.format(str(img_num).zfill(5)))
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
            feed_S = [{}] * self.num_workers 
            feed_G = {}
            for p in range(self.pac_num):
                for w in range(self.num_workers):
                    feed_S[w][self.S[w].Z[p]] = sample[p]
                feed_G[self.G.Z[p]] = sample[p]
            update = [None] * len(self.G.theta_G)
            # update = manager.list([None] * len(self.G.theta_G))
            reward_list = [None] * self.search_num
            reward_list_2 = []
            delta_list = []

            if self.mode == "gradient":
                _, G_loss_curr = self.sess[0].run([self.G_solver, self.G_loss], feed_dict=feed_G)
            else:
                delta = []
                for t in range(len(self.G.theta_G)):
                    delta.append(self.v*np.random.normal(loc=0.0, scale=1.,
                                                size=[self.search_num] + self.G.theta_G[t].get_shape().as_list()))
                for m in range(self.search_num):
                    reward = [None] * self.num_workers
                    for t in range(len(self.G.theta_G)):
                        w = t % self.num_workers
                        i = t // self.num_workers
                        # import pdb; pdb.set_trace()
                        self.sess[w].run(self.update_Sp[w][i], feed_dict={self.delta_ph[w][i]: delta[t][m]})
                    for w in range(self.num_workers):
                        reward[w] = self.sess[w].run(self.S_loss[w], feed_dict=feed_S[w])
                        reward_list_2.append(reward)
                    for t in range(len(self.G.theta_G)):
                        w = t % self.num_workers
                        i = t // self.num_workers
                        self.sess[w].run(self.update_Sn[w][i], feed_dict={self.delta_ph[w][i]: delta[t][m]})
                    for w in range(self.num_workers):
                        tmp = self.sess[w].run(self.S_loss[w], feed_dict=feed_S[w])
                        reward_list_2.append(tmp)
                        reward[w] = reward[w] - tmp

                    for w in range(self.num_workers):
                        for t in self.slices[w]:
                            if m == 0:
                                update[t] = reward[w] * delta[t][m]
                            else:
                                update[t] += reward[w] * delta[t][m]

                # for w in range(self.num_workers):
                #     p = mp.Process(target=search_run, args=(self.sess, self.search_num, self.v, self.S[w], \
                #         self.update_Sp[w], self.update_Sn[w], self.S_loss[w], feed_S[w], self.slices[w], \
                #         self.delta_ph[w], update,))
                #     p.start()
                #     p.join()
                sigma_R = 1
                for w in range(self.num_workers):
                    for t in self.slices[w]:
                        self.sess[w].run(self.update_G[t], feed_dict={self.update_Gph[t]: update[t] * self.alpha / (self.search_num * sigma_R * self.v)})


            G_loss_curr = self.sess[0].run(self.G_loss, feed_dict=feed_G)
            
            for p in range(self.pac_num):
                feed_G[self.X[p]] = X_mb[p]
            _, D_loss_curr, _ = self.sess[0].run([self.D_solver, self.D_loss, self.clip_D], feed_dict=feed_G)

            if it % 10 == 0:
                print('Iter: {}'.format(it))
                # for t in range(1):
                #     print(update[0] * self.alpha / self.search_num)
                #     print(self.sess.run(G.theta_G[0]))
                print('D loss: {:.4}'.format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                # print('Sigma_R: {:.4}'.format(sigma_R))
                writer.writerow({
                    'epoch': it,
                    'D_loss':D_loss_curr,
                    'G_loss':G_loss_curr,
                    'sigma': sigma_R
                })
                print()