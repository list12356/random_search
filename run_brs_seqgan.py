import argparse
from models.brs_seqgan.brs_seqgan import BRSSeqgan


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dir', default='./out_rnn/')
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--l', type=float, default=1.0)
parser.add_argument('--sigma', type=int, default=1)
parser.add_argument('--mode', default="binary")
parser.add_argument('--pac_num', type=int, default=5)
parser.add_argument('--disc', default="RNN")
parser.add_argument('--pre_num', type=int, default=0)
parser.add_argument('--adv_num', type=int, default=15000)
parser.add_argument('--rollout', type=int, default=1)
parser.add_argument('--curriculum', type=int, default=0)
parser.add_argument('--sample', type=int, default=2)
args = parser.parse_args()

out_dir = args.dir
save_step = 1000
search_num = 64
alpha = args.alpha
v = 0.02
_lambda = args.l
mode = args.mode
restore = False
D_lr = 1e-4
pac_num =  args.pac_num
sigma = args.sigma
disc = args.disc
rollout = args.rollout
pre_num = args.pre_num
adv_num = args.adv_num
curriculum = args.curriculum
sample = args.sample

gan = BRSSeqgan(save_dir=out_dir, pre_epoch_num=0, disc_type="RNN", sample_mode=sample,
    search_num = search_num, alpha = alpha, v = v, sigma = sigma, rollout_num = rollout)
gan.init_train(dataset="text")
gan.pretrain(epoch_num=pre_num)
if curriculum != 0:
    for i in range(30):
        gan.train(epoch_num=30, max_len=i + 1, save_dir=out_dir)
else:
    print("fuck")
    gan.train(epoch_num=adv_num)
