import json
import os
from time import time

from models.brs_seqgan.discriminator import CNNDiscriminator
from models.brs_seqgan.discriminator import RNNDiscriminator
from models.brs_seqgan.generator import Generator
from models.brs_seqgan.caption_generator import CaptionGenerator
from utils.text_process import *
from utils.utils import *
from utils.constants import *
from utils.dataloader import *
from nltk.translate.bleu_score import sentence_bleu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg


class BRSSeqgan():
    def __init__(self, oracle=None, save_dir=None,
            pre_epoch_num=0, disc_type="RNN", sample_mode=RNN_SAMPLE_NOISE,
            search_num=64, alpha=0.1, v=0.02, sigma=0, rollout_num=1
            ):
        # you can change parameters, generator here
        self.vocab_size = 20
        self.emb_dim = 256 
        self.hidden_dim = 256
        self.sequence_length = 30
        self.filter_size = [2, 3]
        self.num_filters = [100, 200]
        self.l2_reg_lambda = 0.2
        self.dropout_keep_prob = 0.75
        self.batch_size = 128
        self.generate_num = 128
        self.start_token = 0

        # self.mcmc_num = 1
        # self.prior_std = 1.0
        # self.eta = 2e-4
        # self.alpha = 0.01
        # self.noise_std = np.sqrt(2 * self.alpha * self.eta)
        
        self.img_feature_dim = 0
        self.noise_dim = 100
        self.pre_epoch_num = pre_epoch_num
        self.adversarial_epoch_num = 15000
        self.print_step = 10
        self.save_step = 100
        self.restore_step = 5000
        self.evaluate_step = 1000

        # for language processing
        self.wi_dict = None
        self.iw_dict = None
        self.vocab_filter = 5

        # for gan
        self.searcher = None
        self.generator = None
        self.discriminator = None
        self.gen_data_loader = None
        self.dis_data_loader = None
        self.sess = None
        self.disc_type = disc_type
        self.sample_mode = sample_mode

        # for search
        self.update_Sp = []
        self.update_Sn = []
        self.update_G = []
        self.update_Gph = []
        self.delta_ph = []
        self.reward_ph = None

        # for search prameters
        self.search_num = 64
        self.num_workers = 64
        self.alpha = alpha
        self.v = v
        self.sigma = sigma
        self.rollout_num = rollout_num


        if save_dir == None:
            self.save_dir = './save_fast/'
        else:
            self.save_dir = save_dir
        if os.path.exists(self.save_dir) == False:
            os.makedirs(self.save_dir)
        self.oracle_file = self.save_dir + 'oracle.txt'
        self.truth_file = self.save_dir + 'ground_truth.txt'
        self.generator_file = self.save_dir + 'generator'

    def evaluate(self):
        return
        # test_num = 1000 // 64
        # evaluate_file = self.save_dir + "/evaluate.txt"
        # reference_file = self.save_dir + "reference.txt"
        # score_list = []
        # text_list = []
        # ref_list = []
        # for _ in range(test_num):
        #     img_data, img_meta, text_meta = self.data_loader.next_test_batch()
        #     feed = {self.generator[0].c: img_data}
        #     outputs = self.sess.run(self.generator[0].gen_x, feed_dict = feed)
        #     outputs = outputs.tolist()
        #     score_batch = []
        #     hypo_text = [[self.iw_dict[str(x)] for x in line if x != len(self.iw_dict)] for line in outputs]
        #     # hypo_text = code_to_text(codes=outputs, dictionary=self.iw_dict)
        #     for _ in range(self.batch_size):
        #         ref_text = get_tokenlized(text_meta[_])
        #         text_list.append(' '.join(hypo_text[_]))
        #         ref_list.append(ref_text)
        #         score = sentence_bleu(ref_text, hypo_text[_])
        #         score_batch.append(score)
        #     score = np.mean(score_batch)
        #     print(score)
        #     score_list.append(score)
        # np.mean(score_list)
        # with open(evaluate_file, 'w') as file:
        #     for line in text_list:
        #         file.write(line + '\n')

        # with open(reference_file, 'w') as file:
        #     for line in ref_list:
        #         file.write(' # '.join([' '.join(x) for x in line]))
        #         file.write('\n')

    def init_train(self, dataset="text"):
        if dataset == "flickr":
            self.data_loader = FlickrDataLoader(batch_size=self.batch_size, feature_dim=self.img_feature_dim, seq_length=self.sequence_length)
        elif dataset == "mscoco":
            self.data_loader = MSCOCODataLoader(batch_size=self.batch_size, feature_dim=self.img_feature_dim, seq_length=self.sequence_length)
        elif dataset == "text":
             self.data_loader = TextDataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        else:
            print('wrong dataset!')
            exit(0)
        
        self.wi_dict, self.iw_dict = self.data_loader.create_batches()
        self.vocab_size = len(self.wi_dict) + 1
        
        print('creating model')
        with tf.variable_scope("generator"):
            self.generator = Generator(n_words=self.vocab_size, batch_size=self.batch_size, dim_embed=self.emb_dim,
                                    dim_hidden=self.hidden_dim, sequence_length=self.sequence_length,
                                    dim_image=self.img_feature_dim, dim_noise=self.noise_dim, sample=self.sample_mode)
            self.generator.build_model()
            self.generator.build_generator(self.sequence_length)

        self.searcher =  [None] * self.num_workers
        for w in range(self.num_workers):
            with tf.variable_scope("searcher_" + str(w)):
                self.searcher[w] = Generator(n_words=self.vocab_size, batch_size=self.batch_size, dim_embed=self.emb_dim,
                                        dim_hidden=self.hidden_dim, sequence_length=self.sequence_length,
                                        dim_image=self.img_feature_dim, dim_noise=self.noise_dim, sample=self.sample_mode)
                self.searcher[w].build_model(gradient=False)
                self.searcher[w].build_generator(self.sequence_length)
        with tf.variable_scope("discriminator"):
            if self.disc_type == "CNN":
                self.discriminator = CNNDiscriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                            emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                            l2_reg_lambda=self.l2_reg_lambda)
            if self.disc_type == "RNN":
                self.discriminator = RNNDiscriminator(n_words=self.vocab_size, batch_size=self.batch_size, dim_embed=self.emb_dim,
                                    dim_hidden=self.hidden_dim, sequence_length=self.sequence_length,
                                    dim_image=self.img_feature_dim)

        # build random search
        self.update_Sp = []
        self.update_Sn = []
        self.S_reward_ph = [None] * self.num_workers
        self.S_sample_list = [None] * self.num_workers
        self.S_loss = [None] * self.num_workers
        self.delta_ph = [None] * self.num_workers
        self.update_G = [None] * len(self.generator.theta_G)
        self.reward_ph = tf.placeholder(tf.float32, shape=(self.num_workers))
        self.delta = [None] * len(self.generator.theta_G)
        for w in range(self.num_workers):
            # self.update_Sp[w] = []
            # self.update_Sn[w] = []
            self.delta_ph[w] = []
            self.S_reward_ph[w] = tf.placeholder(tf.int32, shape=(self.batch_size, self.sequence_length))
            # prob = self.discriminator.build(self.S_reward_ph[w])
            prob = self.discriminator.build(self.searcher[w].generated_words)
            self.S_loss[w] = tf.reduce_mean(tf.log(tf.clip_by_value(prob, 1e-16, 1.0)))
            self.S_sample_list[w] = self.searcher[w].generated_words
            for t in range(len(self.generator.theta_G)):
                with tf.device("/device:GPU:0"):
                    delta_ph = tf.placeholder(tf.float32, shape=self.generator.size[t])
                    self.delta_ph[w].append(delta_ph)
                    self.update_Sp.append(tf.assign(self.searcher[w].theta_G[t], self.generator.theta_G[t] + delta_ph, validate_shape=False))
                    self.update_Sn.append(tf.assign(self.searcher[w].theta_G[t], self.generator.theta_G[t] - delta_ph, validate_shape=False))

                    # perturb_shape = self.generator.size[t]
                    # indices = list(range(w, perturb_shape[0], self.num_workers))
                    # perturb_shape[0] = len(indices)
                    # delta_ph = tf.placeholder(tf.float32, shape=perturb_shape)
                    # self.delta_ph[w].append(delta_ph)

                    # self.update_Sp[w].append(tf.scatter_add(self.searcher[w].theta_G[t], indices, delta_ph))
                    # self.update_Sn[w].append(tf.scatter_sub(self.searcher[w].theta_G[t], indices, delta_ph))

        self.delta_ph = np.array(self.delta_ph)
        for t in range(len(self.generator.theta_G)):
            dims = len(self.generator.size[t])
            # TODO: add sigma_R
            update = tf.reduce_sum(tf.reshape(self.reward_ph, [self.num_workers] + [1] * dims) * \
                tf.stack(self.delta_ph[:,t].tolist()), axis=0) * self.alpha / (self.num_workers * self.v)
            self.update_G[t] = tf.assign(self.generator.theta_G[t], self.generator.theta_G[t] + update)
        
        with tf.device("/device:GPU:0"):
            self.update_Sn_op = tf.group(*self.update_Sn)
            self.update_Sp_op = tf.group(*self.update_Sp)
            self.update_G_op = tf.group(*self.update_G)
        
        self.S_sample = tf.stack(self.S_sample_list)
        self.S_reward = tf.stack(self.S_loss, 0)
        print('finished')



        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        self.sess = sess

    def pretrain(self, restore=None, epoch_num=None):
        eof = len(self.wi_dict)
        if epoch_num != None:
            self.pre_epoch_num = epoch_num

        start = time()
        print('start pre-train generator and discriminator:')
        for epoch in range(self.pre_epoch_num * self.data_loader.num_batch):
            img_batch, seq_batch, meta_batch = self.data_loader.next_train_batch()
            current_mask_matrix = np.zeros((seq_batch.shape[0], seq_batch.shape[1]))
            nonzeros = list(map(lambda x: (x != eof).sum() + 1, seq_batch))
            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1
            
            feed = {
                    self.generator.sentence: seq_batch,
                    # self.generator.image: img_batch,
                    self.generator.mask: current_mask_matrix,
                    self.generator.noise: sample_Z(self.batch_size, self.noise_dim)
            }

            _, loss =  self.sess.run([self.generator.pretrain_updates, self.generator.pretrain_loss], feed_dict=feed)
            if epoch % 100 == 0:
                end = time()
                print('epoch_gen:' + str(epoch) + '\t time:' + str(end - start))
                print(loss)
                start = time()

            if epoch % self.evaluate_step ==0:
                self.evaluate()
            if epoch % self.restore_step == 0:
                saver = tf.train.Saver()
                saver.save(self.sess, self.save_dir + "pretrain_epoch" + str(epoch) + ".ckpt")
            if epoch % self.save_step == 0:
                # cg = CaptionGenerator(self.generator)
                # beam = cg.generate_beam(self.sess, np.zeros((self.batch_size, self.img_feature_dim)))
                # samples_beam = [x.sentence for y in beam for x in y]
                # self.draw_samples(self.save_dir + "/beam.txt", samples_beam, seq_batch, meta_batch)
                
                generated_samples = self.sess.run(self.generator.generated_words, feed_dict=feed)
                # generated_samples = np.transpose(np.array(generated_samples))
                self.draw_samples(self.generator_file + str(epoch) + '.txt', generated_samples, seq_batch, meta_batch)



    def train(self, restore=None, epoch_num=None, max_len=None, save_dir=None):
        print("delta")
        self.noise = SharedNoiseTable()
        self.rs = np.random.RandomState()
        print("end")

        eof = len(self.wi_dict)

        self.data_loader.reset_pointer()
        if save_dir == None:
            save_dir = self.save_dir
        if max_len != None:
            self.save_dir = save_dir + '/adv_' + str(max_len) + '/'
        else:
            self.save_dir = self.save_dir + '/adv/'
        if os.path.exists(self.save_dir) == False:
            os.makedirs(self.save_dir)
        self.truth_file = self.save_dir + 'ground_truth.txt'
        self.generator_file = self.save_dir + 'generator'
        
        if epoch_num != None:
            self.adversarial_epoch_num = epoch_num
        if max_len != None:
            self.generator.build_generator(max_len)
            for w in range(self.num_workers):
                self.searcher[w].build_generator(max_len)
        else:
            self.generator.build_generator(self.sequence_length)
            for w in range(self.num_workers):
                self.searcher[w].build_generator(self.sequence_length)
        
        start = time()
        print('adversarial training:')
        for epoch in range(self.adversarial_epoch_num):
            # print('epoch:' + str(epoch))
            start = time()
            img_batch, seq_batch, meta_batch = self.data_loader.next_train_batch()
            # rewards = self.generator.get_reward(self.sess, generated_samples, img_batch, mask, 16, self.discriminator)
            
            reward = np.zeros([self.num_workers])
            reward_list_2 = []
            
            idx = self.noise.sample_index(self.rs, self.noise_dim * self.batch_size)
            sample = np.reshape(self.noise.get(idx, self.noise_dim * self.batch_size), 
                        [self.batch_size, self.noise_dim])
            
            feed = {
                # self.discriminator.image: img_batch,
                # self.generator.image: img_batch,
                self.generator.noise: sample
                }
            for w in range(self.num_workers):
                # feed[self.searcher[w].image] = img_batch
                feed[self.searcher[w].noise] = sample
                
            delta = []
            for t in range(len(self.generator.theta_G)):
                dim = self.num_workers
                for x in self.generator.size[t]:
                    dim = dim * x
                idx = self.noise.sample_index(self.rs, dim)
                delta.append(self.v* np.reshape(self.noise.get(idx, dim),
                    [self.num_workers] + self.generator.size[t]))
            
            feed_update = {}
            for w in range(self.num_workers):
                for t in range(len(self.generator.theta_G)):
                    feed_update[self.delta_ph[w][t]] = delta[t][w]                
                reward_pos = 0

            self.sess.run(self.update_Sp_op, feed_dict=feed_update)
            
            # for w in range(self.num_workers):
            #     for r in range(self.rollout_num):
            #         generated_samples = self.sess.run(self.searcher[w].generated_words, feed_dict=feed)
            #         # generated_samples = np.transpose(np.array(generated_samples))
            #         # current_mask_matrix = np.zeros((generated_samples.shape[0], generated_samples.shape[1]))
            #         # nonzeros = list(map(lambda x: (x != eof).sum() + 1, generated_samples))
            #         # for ind, row in enumerate(current_mask_matrix):
            #         #     if nonzeros[ind] < generated_samples.shape[1]:
            #         #         row[nonzeros[ind]] = 1
            #         #     else:
            #         #         row[generated_samples.shape[1] - 1] = 1
            #         # feed[self.discriminator.mask] = current_mask_matrix
            #         feed[self.discriminator.sentence] = generated_samples
            #         tmp = self.sess.run(self.discriminator.reward, feed_dict=feed)
            #         # reward = self.generator.get_reward(self.sess, 
            #         #     generated_samples, img_batch, current_mask_matrix, 
            #         #     32, self.discriminator)
            #         reward_pos += tmp

            #     reward_pos = reward_pos / self.rollout_num
            #     reward[w] = reward_pos
            #     reward_list_2.append(reward_pos)
            # generated_samples = self.sess.run(self.S_sample, feed_dict=feed)
            # feed_reward = {}
            # for w in range(self.num_workers):
            #     feed_reward[self.S_reward_ph[w]] = generated_samples[w]
            reward = self.sess.run(self.S_reward, feed_dict=feed)

            self.sess.run(self.update_Sn_op, feed_dict=feed_update)
            
            # for w in range(self.num_workers):
            #     reward_neg = 0
            #     for r in range(self.rollout_num):
            #         generated_samples = self.sess.run(self.searcher[w].generated_words, feed_dict=feed)
            #         # generated_samples = np.transpose(np.array(generated_samples))
            #         current_mask_matrix = np.zeros((generated_samples.shape[0], generated_samples.shape[1]))
            #         nonzeros = list(map(lambda x: (x != eof).sum() + 1, generated_samples))
            #         for ind, row in enumerate(current_mask_matrix):
            #             if nonzeros[ind] < generated_samples.shape[1]:
            #                 row[nonzeros[ind]] = 1
            #             else:
            #                 row[generated_samples.shape[1] - 1] = 1
            #         feed[self.discriminator.sentence] = generated_samples
            #         feed[self.discriminator.mask] = current_mask_matrix
            #         tmp = self.sess.run(self.discriminator.reward, feed_dict=feed)
            #         # tmp = self.generator.get_reward(self.sess, 
            #         #     generated_samples, img_batch, current_mask_matrix, 
            #         #     32, self.discriminator)
            #         reward_neg += tmp
                
            #     reward_neg = reward_neg / self.rollout_num
            #     reward_list_2.append(reward_neg)
            #     reward[w] = reward[w] - reward_neg
            
            # generated_samples = self.sess.run(self.S_sample, feed_dict=feed)
            # feed_reward = {}
            # for w in range(self.num_workers):
            #     feed_reward[self.S_reward_ph[w]] = generated_samples[w]
            tmp = self.sess.run(self.S_reward, feed_dict=feed)
            reward = reward - tmp

            if self.sigma == 1:
                sigma_R = np.std(reward_list_2)
            else:
                sigma_R = 1.
            if sigma_R == 0:
                sigma_R = 1.

            feed_update[self.reward_ph] = reward
            self.sess.run(self.update_G_op, feed_dict=feed_update)

            generated_samples = self.sess.run(self.generator.generated_words, feed_dict=feed)
            # generated_samples = np.transpose(np.array(generated_samples))
            current_mask_matrix = np.zeros((generated_samples.shape[0], generated_samples.shape[1]))
            nonzeros = list(map(lambda x: (x != eof).sum() + 1, generated_samples))
            for ind, row in enumerate(current_mask_matrix):
                if nonzeros[ind] < generated_samples.shape[1]:
                    row[nonzeros[ind]] = 1
                else:
                    row[generated_samples.shape[1] - 1] = 1
            feed[self.discriminator.sentence] = generated_samples
            feed[self.discriminator.mask] = current_mask_matrix
            G_loss_curr = self.sess.run(self.discriminator.reward, feed_dict=feed)

            # x, y, c = self.data_loader.disc_batch(generated_samples)
            feed[self.discriminator.truth] = seq_batch
            # real_reward = self.sess.run(self.discriminator.real, feed_dict=feed)
            # feed[self.discriminator.sentence] = x
            # feed[self.discriminator.label] = y
            # feed[self.discriminator.image] = c
            # for i in range(15):
            D_loss_curr,_ = self.sess.run([self.discriminator.d_loss, self.discriminator.train_op], feed_dict=feed)
            
            end = time()

            print('epoch:' + str(epoch) + '\t time:' + str(end - start))
            print('D loss: '+ str(D_loss_curr))
            print('G_loss: ' + str(G_loss_curr))
            # print('Real_reward: ' + str(real_reward))
            print('')
            if epoch % self.save_step == 0 or epoch == self.adversarial_epoch_num - 1:
                generated_samples = self.sess.run(self.generator.generated_words, feed_dict=feed)
                # generated_samples = np.transpose(np.array(generated_samples))
                self.draw_samples(self.generator_file + str(epoch) + '.txt', generated_samples, seq_batch, meta_batch)
                if self.sample_mode == RNN_SAMPLE_ARGMAX:
                    self.generator.sample = RNN_SAMPLE_RANDOM
                    generated_samples = self.sess.run(self.generator.generated_words, feed_dict=feed)
                    # generated_samples = np.transpose(np.array(generated_samples))
                    self.draw_samples(self.generator_file + str(epoch) + '_random.txt', generated_samples, seq_batch, meta_batch)
                    self.generator.sample = RNN_SAMPLE_ARGMAX
                          

    def draw_samples(self, file_name, generated_samples, seq_batch, meta_batch):
        sample_text = None
        true_text = None
        codes = list()
        with open(file_name + '.raw', 'w') as file:
            for line in generated_samples:
                buffer = ' '.join([str(x) for x in line]) + '\n'
                file.write(buffer)
                codes.append(line)
        codes = np.array(codes)
        sample_text = code_to_text(codes=codes, dictionary=self.iw_dict)
        with open(file_name, 'w') as outfile:
            outfile.write(sample_text)
        with open(self.truth_file, 'w') as outfile:
            true_text = code_to_text(codes=seq_batch, dictionary=self.iw_dict)
            outfile.write(true_text)
        sample_text = sample_text.split('\n')
        true_text = true_text.split('\n')
