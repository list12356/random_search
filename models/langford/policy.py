from models.langford.mdp import AntiShape
from models.langford.mdp import ComboLock
from models.langford.mdp import Action
from utils import tf_util as U

import numpy as np
import tensorflow as tf

class Policy:
    def __init__(self, num_hidden=128, num_state=20, horizon=40, batch_size=128):
        self.num_hidden = num_hidden
        self.num_state = num_state
        self.batch_size = batch_size
        self.horizon = horizon
        self.input_state = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.horizon])
        self.input_action = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.horizon])
        self.input_reward = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.horizon])
        self.rollout_state = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])

        with tf.variable_scope("policy"):
            self.W_0 = tf.Variable(tf.random_normal([self.num_state, self.num_hidden]), dtype=tf.float32)
            self.b_0 = tf.Variable(tf.random_normal([self.num_hidden]), dtype=tf.float32)
            self.W_1 = tf.Variable(tf.random_normal([self.num_hidden, 2]), dtype=tf.float32)
            self.b_1 = tf.Variable(tf.random_normal([2]), dtype=tf.float32)
        
        self.params = [self.W_0, self.b_0, self.W_1, self.b_1]

    def build(self):
        # self.gen_state = []
        # self.gen_action = []
        # self.prob = []
        # state = tf.zeros([self.batch_size], dtype=tf.int32)
        # onehot_state = tf.one_hot(state, self.num_state)
        # loss = 0
        # for t in range(self.horizon):
        #     self.gen_state.append(state)
        #     onehot_state = tf.one_hot(state, self.num_state)
        #     h_0 = tf.nn.relu(tf.matmul(onehot_state, self.W_0) + self.b_0)
        #     logit = tf.nn.softmax(tf.matmul(h_0, self.W_1) + self.b_1)
        #     log_prob = tf.log(tf.clip_by_value(logit, 1e-16, 1.0))
        #     action = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
        #     state = tf.clip_by_value(state + 2*action - 1, 0, self.num_state - 1)
        #     self.gen_action.append(action)
        #     self.prob.append(tf.reduce_mean(logit[:,1]))
        
        # self.gen_action = tf.transpose(tf.stack(self.gen_action))
        # self.gen_state = tf.transpose(tf.stack(self.gen_state))
        # self.prob = tf.stack(self.prob)

        onehot_state = tf.one_hot(self.rollout_state, self.num_state)
        h_0 = tf.nn.relu(tf.matmul(onehot_state, self.W_0) + self.b_0)
        logit = tf.nn.softmax(tf.matmul(h_0, self.W_1) + self.b_1)
        log_prob = tf.log(tf.clip_by_value(logit, 1e-16, 1.0))
        self.rollout_action = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)

        loss = 0
        for t in range(self.horizon):
            onehot_state = tf.one_hot(self.input_state[:,t], self.num_state)
            h_0 = tf.nn.relu(tf.matmul(onehot_state, self.W_0) + self.b_0)
            logit = tf.matmul(h_0, self.W_1) + self.b_1
            onehot_label = tf.one_hot(self.input_action[:,t], 2)
            loss += tf.nn.softmax_cross_entropy_with_logits(labels=onehot_label, logits=logit)\
                     * self.input_reward[:, t]
        
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer().minimize(loss)

        self._setfromflat = U.SetFromFlat(self.params)
        self._getflat = U.GetFlat(self.params)

    def set_trainable_flat(self, x):
        self._setfromflat(x)

    def get_trainable_flat(self):
        return self._getflat()

    def roll_out(self, sess, mdp):
        # states, actions = sess.run([self.gen_state, self.gen_action])
        # rewards = np.zeros(shape=[self.batch_size, mdp.horizon])
        # for b in range(self.batch_size):
        #     for s in range(mdp.horizon):
        #         action = Action.go_left
        #         if actions[b][s] == 1:
        #             action = Action.go_right
        #         rewards[b][s] = mdp.get_reward(states[b][s], action)
        #     for s in range(mdp.horizon - 1):
        #         rewards[b][mdp.horizon - s - 2] += rewards[b][mdp.horizon - s - 1]
        gen_state = []
        gen_action = []
        gen_reward = []
        state = np.zeros([self.batch_size], dtype=int)
        for t in range(self.horizon):
            gen_state.append(state)
            actions = sess.run(self.rollout_action, feed_dict={self.rollout_state:state})
            gen_action.append(actions)
            new_state = np.zeros(self.batch_size, dtype=int)
            rewards = []
            for b in range (self.batch_size):
                action = Action.go_left
                if actions[b] == 1:
                    action = Action.go_right
                # SERIOUS BUG HERE
                # sr = mdp.get_reward(state[b], action)
                sr = mdp.dynamics[state[b]][actions[b]]
                new_state[b] = sr.state
                rewards.append(sr.reward)
            state = new_state
            gen_reward.append(rewards)
        
        gen_reward = np.transpose(np.array(gen_reward))
        # for b in range(self.batch_size):
        #     for s in range(mdp.horizon - 1):
        #         gen_reward[b][mdp.horizon - s - 2] += gen_reward[b][mdp.horizon - s - 1]

        return np.transpose(np.array(gen_state)), \
                np.transpose(np.array(gen_action)),\
                gen_reward


class Policy2:
    def __init__(self, num_hidden=128, num_state=20, horizon=40, batch_size=128):
        self.num_hidden = num_hidden
        self.num_state = num_state
        self.batch_size = batch_size
        self.horizon = horizon
        self.input_state = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.horizon])
        self.input_action = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.horizon])
        self.input_reward = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.horizon])
        self.rollout_state = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        self.mask = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.num_state])

        with tf.variable_scope("policy"):
            self.W_0 = tf.Variable(tf.random_normal([self.num_state, self.num_hidden]), dtype=tf.float32)
            self.b_0 = tf.Variable(tf.zeros([self.num_hidden]), dtype=tf.float32)
            self.W_1 = tf.Variable(tf.random_normal([self.num_hidden, self.num_state]), dtype=tf.float32)
            self.b_1 = tf.Variable(tf.zeros([self.num_state]), dtype=tf.float32)
        
        self.params = [self.W_0, self.b_0, self.W_1, self.b_1]

    def build(self):
        onehot_state = tf.one_hot(self.rollout_state, self.num_state)
        h_0 = tf.nn.relu(tf.matmul(onehot_state, self.W_0) + self.b_0)
        self.prob = tf.nn.softmax(tf.matmul(h_0, self.W_1) + self.b_1)
        log_prob = tf.log(tf.clip_by_value(self.prob, 1e-16, 1.0)) * self.mask
        self.rollout_action = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
        
        loss = 0
        for t in range(self.horizon):
            onehot_state = tf.one_hot(self.input_state[:,t], self.num_state)
            h_0 = tf.nn.relu(tf.matmul(onehot_state, self.W_0) + self.b_0)
            logit = tf.matmul(h_0, self.W_1) + self.b_1
            onehot_label = tf.one_hot(self.input_action[:,t], self.num_state)
            loss += tf.nn.softmax_cross_entropy_with_logits(labels=onehot_label, logits=logit)\
                     * self.input_reward[:, t]
        
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer().minimize(loss)

        self._setfromflat = U.SetFromFlat(self.params)
        self._getflat = U.GetFlat(self.params)


    def set_trainable_flat(self, x):
        self._setfromflat(x)

    def get_trainable_flat(self):
        return self._getflat()

    def roll_out(self, sess, mdp):
        gen_state = []
        gen_action = []
        gen_reward = []
        state = np.zeros([self.batch_size], dtype=int)
        for t in range(self.horizon):
            gen_state.append(state)
            # mask = np.zeros([self.batch_size, self.num_state])
            # for b in range(self.batch_size):
            #     for action in mdp.get_action(state[b]):
            #         mask[b][action.state] = 1
            # actions = sess.run(self.rollout_action, feed_dict={self.mask: mask})
            actions = np.zeros(self.batch_size, dtype=int)
            prob = sess.run(self.prob, feed_dict={self.rollout_state:state})
            rewards = []
            for b in range(self.batch_size):
                prob_normalized = []
                action_list = mdp.get_action(state[b])
                for action in action_list:
                    prob_normalized.append(prob[b][action.state])
                prob_normalized = np.array(prob_normalized)
                prob_normalized = prob_normalized / np.sum(prob_normalized)
                action = np.random.choice(len(action_list), 1, p=prob_normalized)
                actions[b] = action_list[action[0]].state
                sr = mdp.get_reward(state[b], action)
                rewards.append(sr.reward)
            gen_action.append(actions)
            state = actions
            gen_reward.append(rewards)
        
            # import pdb; pdb.set_trace()
        
        gen_reward = np.transpose(np.array(gen_reward))
        for b in range(self.batch_size):
            for s in range(mdp.horizon - 1):
                gen_reward[b][mdp.horizon - s - 2] += gen_reward[b][mdp.horizon - s - 1]

        return np.transpose(np.array(gen_state)), \
                np.transpose(np.array(gen_action)),\
                gen_reward

class Policy3:
    def __init__(self, num_hidden=128, num_state=20, horizon=40, batch_size=128):
        self.num_hidden = num_hidden
        self.num_state = num_state
        self.batch_size = batch_size
        self.horizon = horizon
        self.input_state = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.horizon])
        self.input_action = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.horizon])
        self.input_reward = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.horizon])
        self.rollout_state = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        self.rollout_noise = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 2])

        with tf.variable_scope("policy"):
            self.W_0 = tf.Variable(tf.random_normal([self.num_state, self.num_hidden]), dtype=tf.float32)
            self.b_0 = tf.Variable(tf.random_normal([self.num_hidden]), dtype=tf.float32)
            self.W_1 = tf.Variable(tf.random_normal([self.num_hidden, 2]), dtype=tf.float32)
            self.b_1 = tf.Variable(tf.random_normal([2]), dtype=tf.float32)
        
        self.params = [self.W_0, self.b_0, self.W_1, self.b_1]

    def build(self):
        onehot_state = tf.one_hot(self.rollout_state, self.num_state)
        h_0 = tf.nn.relu(tf.matmul(onehot_state, self.W_0) + self.b_0)
        logit = tf.nn.softmax(tf.matmul(h_0, self.W_1) + self.b_1)
        log_prob = tf.log(tf.clip_by_value(logit, 1e-16, 1.0))
        self.rollout_action = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
        self.rollout_action_noise = tf.cast(tf.reshape(tf.multinomial(log_prob + self.rollout_noise, 1), [self.batch_size]), tf.int32)
        loss = 0
        for t in range(self.horizon):
            onehot_state = tf.one_hot(self.input_state[:,t], self.num_state)
            h_0 = tf.nn.relu(tf.matmul(onehot_state, self.W_0) + self.b_0)
            logit = tf.matmul(h_0, self.W_1) + self.b_1
            onehot_label = tf.one_hot(self.input_action[:,t], 2)
            loss += tf.nn.softmax_cross_entropy_with_logits(labels=onehot_label, logits=logit)\
                     * self.input_reward[:, t]
        
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer().minimize(loss)

        self._setfromflat = U.SetFromFlat(self.params)
        self._getflat = U.GetFlat(self.params)

    def set_trainable_flat(self, x):
        self._setfromflat(x)

    def get_trainable_flat(self):
        return self._getflat()

    def roll_out(self, sess, mdp):
        gen_state = []
        gen_action = []
        gen_reward = []
        state = np.zeros([self.batch_size], dtype=int)
        for t in range(self.horizon):
            gen_state.append(state)
            actions = sess.run(self.rollout_action, feed_dict={self.rollout_state:state})
            gen_action.append(actions)
            new_state = np.zeros(self.batch_size, dtype=int)
            rewards = []
            for b in range (self.batch_size):
                action = Action.go_left
                if actions[b] == 1:
                    action = Action.go_right
                # SERIOUS BUG HERE
                # sr = mdp.get_reward(state[b], action)
                sr = mdp.dynamics[state[b]][actions[b]]
                new_state[b] = sr.state
                rewards.append(sr.reward)
            state = new_state
            gen_reward.append(rewards)
        
        gen_reward = np.transpose(np.array(gen_reward))
        # for b in range(self.batch_size):
        #     for s in range(mdp.horizon - 1):
        #         gen_reward[b][mdp.horizon - s - 2] += gen_reward[b][mdp.horizon - s - 1]

        return np.transpose(np.array(gen_state)), \
                np.transpose(np.array(gen_action)),\
                gen_reward

    def roll_out_with_noise(self, sess, mdp, noise):
        gen_state = []
        gen_action = []
        gen_reward = []
        state = np.zeros([self.batch_size], dtype=int)
        for t in range(self.horizon):
            gen_state.append(state)
            actions = sess.run(self.rollout_action_noise, \
                feed_dict={
                    self.rollout_state:state,
                    self.rollout_noise:noise[t]
                })
            gen_action.append(actions)
            new_state = np.zeros(self.batch_size, dtype=int)
            rewards = []
            for b in range (self.batch_size):
                action = Action.go_left
                if actions[b] == 1:
                    action = Action.go_right
                # SERIOUS BUG HERE
                # sr = mdp.get_reward(state[b], action)
                sr = mdp.dynamics[state[b]][actions[b]]
                new_state[b] = sr.state
                rewards.append(sr.reward)
            state = new_state
            gen_reward.append(rewards)
        
        gen_reward = np.transpose(np.array(gen_reward))
        # for b in range(self.batch_size):
        #     for s in range(mdp.horizon - 1):
        #         gen_reward[b][mdp.horizon - s - 2] += gen_reward[b][mdp.horizon - s - 1]

        return np.transpose(np.array(gen_state)), \
                np.transpose(np.array(gen_action)),\
                gen_reward