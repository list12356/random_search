import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import numpy as np
from utils.constants import *
from time import time

class Generator(object):
    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)

    def __init__(self, dim_image, dim_embed, dim_hidden, batch_size,
         sequence_length, n_words, bias_init_vector=None, start_token=0, 
         sample=RNN_SAMPLE_RANDOM, dim_noise=100):

        self.dim_image = np.int(dim_image)
        self.dim_embed = np.int(dim_embed)
        self.dim_hidden = np.int(dim_hidden)
        self.dim_noise = np.int(dim_noise)
        self.batch_size = np.int(batch_size)
        self.sequence_length = np.int(sequence_length)
        self.n_words = np.int(n_words)
        self.bias_init_vector = bias_init_vector
        self.learning_rate = 0.001
        self.beam_size = 3
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        
        # Parameters
        self.Wemb = None
        self.bemb = None
        self.lstm = None
        self.encode_img_W = None
        self.encode_img_b = None
        self.encode_noise_W = None
        self.encode_noise_b = None
        self.embed_word_W = None
        self.embed_word_b =None

        self.image = None
        self.sentence = None
        self.mask = None
        self.reward = None
        self.image_emb = None
        self.pretrain_loss = None
        self.pretrain_updates = None
        self.loss = None
        self.adv_updates = None

        self.generated_words = []
        self.generated_words_2 = None
        self.given_num = None
        self.sample = sample 
        self.noise = None

        # used for beam search
        self.state_feed = None
        self.word_feed = None
        self.infer_word = None
        self.infer_state = None

        # for random search
        self.search_num = 32
        self.theta_G = None
        self.size = None
        self.update_Sn = None
        self.update_Sp = None
        self.update_g_ops = None
        self.delta_ph = None
        self.update_ph = None

        # self.build_model()
        # self.build_generator(self.sequence_length)
        # self.build_reward()
        # self.build_beam()


    def build_model(self, gradient=True):
        
        self.noise = tf.placeholder(tf.float32, [self.batch_size, self.dim_noise])
        self.sentence = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length])
        self.mask = tf.placeholder(tf.float32, [self.batch_size, self.sequence_length])
        self.reward = tf.placeholder(tf.float32, [self.batch_size, self.sequence_length])

        with tf.variable_scope("embed") as vs_emb:
            with tf.device("/cpu:0"):
                self.Wemb = tf.Variable(tf.random_uniform([self.n_words, self.dim_embed], -0.1, 0.1), name='Wemb')

            self.bemb = self.init_bias(self.dim_embed, name='bemb')

            self.lstm = tf.nn.rnn_cell.BasicLSTMCell(self.dim_hidden)
            
            if self.dim_image > 0:
                self.image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])
                self.encode_img_b = self.init_bias(self.dim_hidden, name='encode_img_b')
                if self.sample == RNN_SAMPLE_NOISE:
                    self.encode_img_W = tf.Variable(tf.random_uniform(
                        [self.dim_image + self.dim_noise, self.dim_hidden], -0.1, 0.1), name='encode_img_W')
                    # (batch_size, dim_hidden)
                    self.image_emb = tf.matmul(tf.concat([self.image, self.noise], 1),
                        self.encode_img_W) + self.encode_img_b 
                else:
                    self.encode_img_W = tf.Variable(tf.random_uniform(
                        [self.dim_image, self.dim_hidden], -0.1, 0.1), name='encode_img_W')
                    # (batch_size, dim_hidden)
                    self.image_emb = tf.matmul(self.image, self.encode_img_W) + self.encode_img_b
            else:
                if self.sample == RNN_SAMPLE_NOISE:
                    self.encode_noise_b = self.init_bias(self.dim_hidden, name='encode_noise_b')
                    self.encode_noise_W = tf.Variable(tf.random_uniform(
                        [self.dim_noise, self.dim_hidden], -0.1, 0.1), name='encode_noise_W')
                    # (batch_size, dim_hidden)
                    self.image_emb = tf.matmul(self.noise, self.encode_noise_W) + self.encode_noise_b
                else:
                    self.image_emb = tf.zeros(shape=[self.batch_size, self.dim_hidden])
                

            self.embed_word_W = tf.Variable(tf.random_uniform([self.dim_hidden, self.n_words], -0.1, 0.1), name='embed_word_W')

            if self.bias_init_vector is not None:
                self.embed_word_b = tf.Variable(self.bias_init_vector.astype(np.float32), name='embed_word_b')
            else:
                self.embed_word_b = self.init_bias(self.n_words, name='embed_word_b')
            
        self.theta_G = [v for v in tf.all_variables() if v.name.startswith(vs_emb.name)]
        

        # define model for MLE training
        if gradient ==True:
        
            state = self.lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)

            loss = 0.0
            adv_loss = 0.0
            with tf.variable_scope("RNN") as vs_lstm:
                current_emb = self.image_emb
                output, state = self.lstm(current_emb, state) # (batch_size, dim_hidden)
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    for i in range(1, self.sequence_length): # maxlen + 1
                        with tf.device("/cpu:0"):
                            current_emb = tf.nn.embedding_lookup(self.Wemb, self.sentence[:,i-1]) + self.bemb
                        output, state = self.lstm(current_emb, state) # (batch_size, dim_hidden)
                        labels = tf.expand_dims(self.sentence[:, i], 1) # (batch_size)
                        indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                        concated = tf.concat([indices, labels], 1)
                        onehot_labels = tf.sparse_to_dense(
                                concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0) # (batch_size, n_words)

                        logit_words = tf.matmul(output, self.embed_word_W) + self.embed_word_b # (batch_size, n_words)
                        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
                        cross_entropy = cross_entropy * self.mask[:,i]#tf.expand_dims(self.mask, 1)

                        current_loss = tf.reduce_sum(cross_entropy)
                        adv_loss = adv_loss + current_loss*self.reward[i]
                        loss = loss + current_loss

            for v in tf.all_variables():
                if v.name.startswith(vs_lstm.name):
                    self.theta_G.append(v)

            self.size = [self.theta_G[t].get_shape().as_list() for t in range(len(self.theta_G))]
            self.pretrain_loss = loss / tf.reduce_sum(self.mask[:,1:])
            self.pretrain_updates = tf.train.AdamOptimizer(self.learning_rate).minimize(self.pretrain_loss, var_list=self.theta_G)
            self.loss = adv_loss / tf.reduce_sum(self.mask[:,1:])
            self.adv_updates = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
        else:
            state = self.lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            with tf.variable_scope("RNN") as vs_lstm:
                current_emb = self.image_emb
                output, state = self.lstm(current_emb, state)

            for v in tf.all_variables():
                if v.name.startswith(vs_lstm.name):
                    self.theta_G.append(v)

    def build_beam(self):
        self.state_feed = tf.placeholder(dtype=tf.float32,
                                    shape=[None, sum(self.lstm.state_size)],
                                    name="state_feed")
        self.word_feed = tf.placeholder(dtype=tf.int32,
                                    shape=[None],
                                    name="word_feed")
        # with tf.variable_scope("beam_step"):
        state_tuple = tf.split(value=self.state_feed, num_or_size_splits=2, axis=1)
        last_word = tf.nn.embedding_lookup(self.Wemb, self.word_feed) + self.bemb
        output, state_tuple = self.lstm(last_word, state_tuple)
        infer_logitis = tf.nn.softmax(tf.matmul(output, self.embed_word_W) + self.embed_word_b)
        # self.infer_output = tf.log(infer_logitis)
        self.infer_state = tf.concat(axis=1, values=state_tuple)
        self.infer_output = infer_logitis
            # (batch_size x n_words)


    def build_generator(self, maxlen):
        self.generated_words = []
        state = self.lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        with tf.variable_scope("RNN"):
            output, state = self.lstm(self.image_emb, state)
            last_word = tf.nn.embedding_lookup(self.Wemb, self.start_token) + self.bemb

            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                for i in range(maxlen):
                    output, state = self.lstm(last_word, state)

                    logit_words = tf.matmul(output, self.embed_word_W) + self.embed_word_b
                    # prevent trap into start token
                    logit_words = logit_words[:, 1:self.n_words]
                    # (batch_size x n_words)
                    if self.sample == RNN_SAMPLE_RANDOM:
                        log_prob = tf.log(tf.nn.softmax(logit_words))
                        max_prob_word = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
                    else:
                        max_prob_word = tf.argmax(logit_words, 1)
                    # shift the start token since we removed it previously
                    max_prob_word += 1
                    with tf.device("/cpu:0"):
                        last_word = tf.nn.embedding_lookup(self.Wemb, max_prob_word)
                    last_word += self.bemb

                    self.generated_words.append(max_prob_word)

        for i in range(self.sequence_length - maxlen):
            self.generated_words.append(tf.ones([self.batch_size], dtype=tf.int32) * (self.n_words - 1))
        # batch x seq_len
        self.generated_words = tf.transpose(tf.stack(self.generated_words))

    def build_reward(self):
        state = self.lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        self.given_num = tf.placeholder(tf.int32)
        #last_word = self.image_emb #
        self.generated_words_2 = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        with tf.variable_scope("Reward"):
            output, state = self.lstm(self.image_emb, state)
            # last_word = tf.nn.embedding_lookup(self.Wemb, [0]) + self.bemb

            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                def _g_recurrence_1(i, state, given_num, generated_words_2):
                    with tf.device("/cpu:0"):
                       current_emb = tf.nn.embedding_lookup(self.Wemb, self.sentence[:,i-1]) + self.bemb
                    output, state = self.lstm(current_emb, state)
                    logit_words = tf.matmul(output, self.embed_word_W) + self.embed_word_b
                    max_prob_word = tf.argmax(logit_words, 1)
                    generated_words_2.write(i - 1, max_prob_word)
                    return i + 1, state, given_num, generated_words_2

                # When current index i >= given_num, start roll-out, use the output as time step t as the input at time step t+1
                def _g_recurrence_2(i, last_word, state, generated_words_2):
                    output, state = self.lstm(last_word, state)

                    logit_words = tf.matmul(output, self.embed_word_W) + self.embed_word_b
                    max_prob_word = tf.argmax(logit_words, 1)

                    with tf.device("/cpu:0"):
                        last_word = tf.nn.embedding_lookup(self.Wemb, max_prob_word)
                    last_word += self.bemb

                    generated_words_2.write(i, max_prob_word)
                    return i + 1, last_word, state, generated_words_2

                i, state, _, self.generated_words_2 = control_flow_ops.while_loop(
                    cond=lambda i, _1, given_num, _4: i <= given_num,
                    body=_g_recurrence_1,
                    loop_vars=(tf.constant(1, dtype=tf.int32),
                               state, self.given_num, self.generated_words_2))
                
                with tf.device("/cpu:0"):
                    last_word = tf.nn.embedding_lookup(self.Wemb, self.generated_words_2.read(i - 2))
                last_word += self.bemb
                _, _, _, self.generated_words_2 = control_flow_ops.while_loop(
                    cond=lambda i, _1, _2, _3: i < self.sequence_length,
                    body=_g_recurrence_2,
                    loop_vars=(i, last_word, state, self.generated_words_2))
                self.generated_words_2 = self.generated_words_2.stack()

    def get_trainop(self):
        return
        # return tf.train.AdamOptimizer(self.learning_rate).minimize(self.pretrain_loss)


    def get_reward(self, sess, sentence, image, mask, rollout_num, discriminator):
        rewards = []
        for i in range(rollout_num):
            for given_num in range(1, len(sentence[0])): #sequence_length
                feed = {self.sentence: sentence, self.image: image, self.mask: mask, self.given_num: given_num}
                samples = sess.run(self.generated_words_2, feed_dict=feed)
                samples = np.transpose(np.array(samples))
                feed = {discriminator.sentence: samples, discriminator.image: image}
                ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            # the last token reward
            feed = {discriminator.sentence: sentence, discriminator.image: image}
            ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[(len(sentence[0])-1)] += ypred

        reward_res = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
        return reward_res        
