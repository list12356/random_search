from models.note_rnn import rl_tuner_eval_metrics as metric
from models.note_rnn.evaluator import Evaluator
from multiprocessing import Pool
import random
import numpy as np
import time
import os


sequence_length = 32
num_actions = 38

batch_size = 64
worker_num = 16

def calc_reward(piece):
    e = Evaluator()
    e.reset_composition()
    reward = [None] * sequence_length
    for j in range(sequence_length):
        obs_note = piece[j]
        new_observation = np.eye(num_actions)[obs_note]
        reward[j] = e.reward_music_theory(new_observation)
        e.composition.append(np.argmax(new_observation))
        e.beat += 1
    return reward

d = metric.initialize_stat_dict()
# e = Evaluator()
rewards = []
pieces = []
for i in range(worker_num*batch_size):
    pieces.append([random.randint(0, num_actions - 1) for _ in range(sequence_length)])
start = time.time()
for piece in pieces:
    reward = calc_reward(piece)
    rewards.append(reward)
end = time.time()
print('time' + str(end - start))
# s = metric.get_stat_dict_string(d)
# print(s)

rewards_async = []
start = time.time()
pool = Pool(os.cpu_count())
result = list()
for piece in pieces:
    result.append(pool.apply_async(calc_reward, args=(piece, )))
for i in result:
    rewards_async.append(i.get())
pool.close()
pool.join()
end = time.time()
print('time' + str(end - start))
import pdb; pdb.set_trace()
