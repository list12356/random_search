from models.langford.mdp import AntiShape
from models.langford.mdp import ComboLock
from models.langford.mdp import Action
from models.langford.policy import Policy
from models.langford.policy import Policy2
from models.langford.policy import Policy3
from utils.utils import SharedNoiseTable
from time import time

import tensorflow as tf
import numpy as np

# parameter for the mdp
num_state = 10
max_perturb_step = 5
mdp = ComboLock()
print("random controller on antishape")
mdp.create(num_state)
train_epoch = 1000
batch_size = 128
epsilon = 1e-3
search_num = 16
v = 0.02
alpha = 1e-4

noise = SharedNoiseTable()
rs = np.random.RandomState()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession()

log_file = open('./test_log.csv', 'w+')
plot_file = open('./test_log.csv', 'w+')

start = time()
reward_list = []
for run in range(20):
    print ("run: " + str(run))
    policy = Policy3(num_state=num_state, horizon=mdp.horizon, batch_size=batch_size)
    policy.build()
    sess.run(tf.global_variables_initializer())
    states_list = []
    noise_step = 0 
    delta = []
    perturb_step = 0
    for t in range(mdp.horizon):
        delta.append(np.zeros(shape=(batch_size, 2)))
    for epoch in range(train_epoch):
        if noise_step > 0:
            states, actions, rewards = policy.roll_out_with_noise(sess, mdp, delta)
            noise_step -= 1
        else:
            states, actions, rewards = policy.roll_out(sess, mdp)
        feed = {
            policy.input_action : actions,
            policy.input_state : states,
            policy.input_reward : rewards
        }
        states_list.append(states)
        reward_list.append(np.sum(np.mean(rewards, axis=0)))
        if len(reward_list) > 5:
            reward_list.pop(0)

        _, loss = sess.run([policy.train_op, policy.loss], feed_dict=feed)
        if epoch % 10 == 0:
            # import pdb; pdb.set_trace()
            print("epoch:" + str(epoch))
            print(reward_list[-1])
            print(np.var(reward_list))
            
            if len(reward_list) == 5 and np.var(reward_list) < epsilon and noise_step == 0:
                print('perturbing the graph')
                dim = batch_size*2
                idx = noise.sample_index(rs, dim)
                # delta[perturb_step] = np.reshape(noise.get(idx, dim), (batch_size, 2))
                delta[perturb_step] = np.random.normal(loc=0.0, scale=10., size=(batch_size, 2))
                noise_step += 20
                perturb_step += 1
                if perturb_step > max_perturb_step:
                    perturb_step = 0
                    noise_step = 0
                # for t in range(mdp.horizon):
                #     delta[t] = delta[perturb_step] = np.random.normal(loc=0.0, scale=10., size=(batch_size, 2))
                # for t in range(mdp.horizon):
                #     delta[t] = np.random.exponential(scale=1., size=((batch_size, 2)))
                # reward_max = np.max(reward_list)
                # states, actions, rewards = policy.roll_out_with_noise(sess, mdp, delta)
                # reward = np.sum(np.mean(rewards, axis=0))

    buff = np.bincount(np.reshape(np.concatenate(states_list), [-1]))
    for x in buff:
        log_file.write(str(x) + ',')
    for w in reward_list:
        log_file.write(str(w) + ',')
    log_file.write('\n')
    log_file.flush()
    # import pdb; pdb.set_trace()
    # log_file.write(np.sum(np.concatenate(states_list), axis=0))
