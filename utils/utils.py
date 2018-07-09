import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import os

from utils.text_process import code_to_text

def init_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    return sess

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def plot_mnist(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='gray')

    return fig

def sample_Z(m, n):
    return np.random.randn(m, n)

def draw_samples(save_dir, epoch, generated_samples, seq_batch, meta_batch, iw_dict, file_name = ""):
        sample_text = None
        true_text = None
        codes = list()

        if os.path.exists(save_dir + '/' + str(epoch)) == False:
            os.makedirs(save_dir + '/' + str(epoch))
        
        with open(save_dir + '/' + str(epoch) + '/' + file_name + 'raw.txt', 'w') as file:
            for line in generated_samples:
                buffer = ' '.join([str(x) for x in line]) + '\n'
                file.write(buffer)
                codes.append(line)
        
        codes = np.array(codes)
        sample_text = code_to_text(codes=codes, dictionary=iw_dict)
        true_text = code_to_text(codes=seq_batch, dictionary=iw_dict)
        
        with open(save_dir + '/' + str(epoch) + '/' + file_name + 'text.txt', 'w') as outfile:
            outfile.write(sample_text)
            outfile.flush()
        
        with open(save_dir + '/' + str(epoch) + '/truth.txt', 'w') as outfile:
            outfile.write(true_text)
            outfile.flush()
        
        sample_text = sample_text.split('\n')
        true_text = true_text.split('\n')
        
        for i in range(4): # ad-hoc
            fig = plt.figure(figsize=(8, 8))
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)
            for j in range(16):
                ax = plt.subplot(gs[j])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(mpimg.imread(meta_batch[i * 16 + j]))
        
            plt.savefig(save_dir + '/' + str(epoch) + '/' + str(i) + ".png", bbox_inches="tight")
            plt.close(fig)


