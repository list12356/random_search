from models.langford.mdp import AntiShape
from models.langford.mdp import ComboLock
from models.langford.mdp import Action
from models.langford.policy import Policy
from models.langford.policy import Policy2
from utils.utils import SharedNoiseTable
from time import time

import tensorflow as tf
import numpy as np

num_state = 10
mdp = AntiShape()
print("random controller on antishape")
mdp.create(num_state)
train_epoch = 100
batch_size = 128
epsilon = 1e-3
search_num = 16
v = 0.02
alpha = 0.2

noise = SharedNoiseTable()
rs = np.random.RandomState()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession()

log_file = open('./test_log.csv', 'w+')
reward_log_file = open('./test_reward.csv', 'w+')
loss_log_file = open('./test_loss.csv', 'w+')

start = time()
reward_list = []
for run in range(20):
    print ("run: " + str(run))
    policy = Policy(num_state=num_state, horizon=mdp.horizon, batch_size=batch_size)
    policy.build()
    sess.run(tf.global_variables_initializer())
    states_list = []
    for epoch in range(train_epoch):
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

        loss_log_file.write(str(loss) + ',')
        reward_log_file.write(str(np.sum(np.mean(rewards, axis=0))) + ',')
        if epoch % 10 == 0:
            # import pdb; pdb.set_trace()
            print("epoch:" + str(epoch))
            print(reward_list[-1])
            print(np.var(reward_list))

            if len(reward_list) == 5 and np.var(reward_list) < epsilon:
                print('perturbing the parameters')
                theta = policy.get_trainable_flat()
                dim = np.shape(theta)[0]
                search_reward = []
                delta_dix = []
                reward_max = np.max(reward_list)
                theta_update = 0
                available_direction = 0
                for w in range(search_num):
                    idx = noise.sample_index(rs, dim)
                    delta = noise.get(idx, dim)
                    theta_plus = theta + delta
                    theta_minus = theta - delta
                    
                    policy.set_trainable_flat(theta_plus)
                    states, actions, rewards = policy.roll_out(sess, mdp)
                    reward_plus = np.sum(np.mean(rewards, axis=0))

                    policy.set_trainable_flat(theta_minus)
                    states, actions, rewards = policy.roll_out(sess, mdp)
                    reward_minus = np.sum(np.mean(rewards, axis=0))

                    delta_dix.append(idx)
                    search_reward.append(reward_plus)
                    search_reward.append(reward_minus)
                    if reward_plus > reward_max + 5 or reward_minus > reward_max + 5:
                        available_direction += 1
                        theta_update += (reward_plus - reward_minus) * delta

                if available_direction > 0:
                    theta_update = theta_update/ (v * available_direction)
                    # may use adam instead
                    theta_update = theta + theta_update * alpha
                    policy.set_trainable_flat(theta_update)
                else:
                    # import pdb; pdb.set_trace()
                    print('No available direction')
                    policy.set_trainable_flat(theta)

    buff = np.bincount(np.reshape(np.concatenate(states_list), [-1]))
    for x in buff:
        log_file.write(str(x) + ',')
    for w in reward_list:
        log_file.write(str(w) + ',')
    log_file.write('\n')
    loss_log_file.write('\n')
    reward_log_file.write('\n')
    log_file.flush()
    loss_log_file.flush()
    reward_log_file.flush()
    # import pdb; pdb.set_trace()
    # log_file.write(np.sum(np.concatenate(states_list), axis=0))
