import tensorflow as tf
import numpy as np

def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


class CNNDiscriminator(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            emd_dim, filter_sizes, num_filters, img_dim=4096, l2_reg_lambda=0.0, dropout_keep_prob = 1, noise_std=2e-3):
        # Placeholders for input, output and dropout
        self.sentence = tf.placeholder(tf.int32, [None, sequence_length], name="sentence")
        self.label = tf.placeholder(tf.float32, [None, num_classes], name="label")
        self.image = tf.placeholder(tf.float32, [None, img_dim], name="image")
        self.mask = tf.placeholder(tf.float32, [None, sequence_length])
        self.dropout_keep_prob = dropout_keep_prob
        self.prior_std = 1.0
        self.noise_std = noise_std
        # self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.variable_scope('CNN'):
            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, emd_dim], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.sentence)
                # self.embedded_joint = tf.concat([self.embedded_chars, self.image], axis=1)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, emd_dim, 1, num_filter]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = sum(num_filters)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add highway
            with tf.name_scope("highway"):
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

            # Make the dot prod here
            # import pdb; pdb.set_trace()

            if img_dim > 0:
                self.h_drop = tf.concat([self.image, self.h_drop], axis = 1)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                if img_dim > 0:
                    W = tf.Variable(tf.truncated_normal([num_filters_total + img_dim, num_classes], stddev=0.1), name="W")
                else:
                    W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.ypred_for_auc = tf.nn.softmax(self.scores)
                self.reward = tf.reduce_mean(tf.log(tf.clip_by_value(self.ypred_for_auc, 1e-16, 1.0)), axis=0)
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.label)
                self.pretrain_loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
                self.d_loss = tf.reshape(tf.reduce_mean(self.loss), shape=[1])
                self.pretrain_d_loss = tf.reshape(tf.reduce_mean(self.pretrain_loss), shape=[1])

        self.params = [param for param in tf.trainable_variables() if 'CNN' in param.name]
        d_optimizer = tf.train.AdamOptimizer(1e-4)
        d_optimizer_mom = tf.train.MomentumOptimizer(1e-4, 0.01)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        pretrain_grad = d_optimizer.compute_gradients(self.pretrain_loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)
        self.train_op_mom = d_optimizer_mom.apply_gradients(grads_and_vars)
        self.pretrain_op = d_optimizer.apply_gradients(pretrain_grad)


class RNNDiscriminator(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)
    
    def __init__(self, dim_image, dim_embed, dim_hidden, batch_size,
         sequence_length, n_words, bias_init_vector=None, start_token=0):
        # Placeholders for input, output and dropout
        self.dim_image = np.int(dim_image)
        self.dim_embed = np.int(dim_embed)
        self.dim_hidden = np.int(dim_hidden)
        self.batch_size = np.int(batch_size)
        self.sequence_length = np.int(sequence_length)
        self.n_words = np.int(n_words)
        self.bias_init_vector = bias_init_vector
        self.learning_rate = 0.001
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        
        # Parameters
        self.Wemb = None
        self.bemb = None
        self.lstm = None
        self.encode_img_W = None
        self.encode_img_b = None
        self.embed_word_W = None
        self.embed_word_b =None
        self.fc_W = None
        self.fc_b = None

        self.image = None
        self.sentence = None
        self.label = None
        self.mask = None
        self.reward = None
        self.image_emb = None
        self.d_loss = None
        self.reward = None

        self.generated_words = []
        self.generated_words_2 = None
        self.given_num = None

        self.reuse = False

        self.build_model()
        self.build_disc()


    def build_model(self):
        
        class_num = 1
        with tf.variable_scope("embed") as vs_emb:
            with tf.device("/cpu:0"):
                self.Wemb = tf.Variable(tf.random_uniform([self.n_words, self.dim_embed], -0.1, 0.1), name='Wemb')

            self.bemb = self.init_bias(self.dim_embed, name='bemb')

            self.lstm = tf.nn.rnn_cell.BasicLSTMCell(self.dim_hidden)

            if self.dim_image > 0:
                self.encode_img_W = tf.Variable(tf.random_uniform([self.dim_image, self.dim_hidden], -0.1, 0.1), name='encode_img_W')
                self.encode_img_b = self.init_bias(self.dim_hidden, name='encode_img_b')
                self.image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])
                self.image_emb = tf.matmul(self.image, self.encode_img_W) + self.encode_img_b # (batch_size, dim_hidden)
            else:
                self.image_emb = tf.zeros(shape=[self.batch_size, self.dim_hidden])

            self.embed_word_W = tf.Variable(tf.random_uniform([self.dim_hidden, self.n_words], -0.1, 0.1), name='embed_word_W')

            if self.bias_init_vector is not None:
                self.embed_word_b = tf.Variable(self.bias_init_vector.astype(np.float32), name='embed_word_b')
            else:
                self.embed_word_b = self.init_bias(self.n_words, name='embed_word_b')

            self.sentence = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length])
            self.truth = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length])
            self.mask = tf.placeholder(tf.float32, [self.batch_size, self.sequence_length])

            
            self.fc_W = tf.Variable(tf.random_uniform([self.dim_hidden, class_num], -0.1, 0.1), name="fc_W")
            self.fc_b = self.init_bias(class_num, name = "fc_b")

            self.label = tf.placeholder(tf.float32, [self.batch_size, 2], name="label")
            
        # self.theta_G = [v for v in tf.all_variables() if v.name.startswith(vs_emb.name)]
        # import pdb; pdb.set_trace()
    
    def build(self, X):
        state = self.lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        final_state = tf.zeros(shape = [self.batch_size, self.dim_hidden], dtype=tf.float32)
        prob = []
        with tf.variable_scope("RNN", reuse=self.reuse) as vs_lstm:
            output, state = self.lstm(self.image_emb, state)
            last_word = tf.nn.embedding_lookup(self.Wemb, self.start_token) + self.bemb

            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                for i in range(self.sequence_length):
                    with tf.device("/cpu:0"):
                        current_emb = tf.nn.embedding_lookup(self.Wemb, X[:,i-1]) + self.bemb
                    output, state = self.lstm(current_emb, state) # (batch_size, dim_hidden)
                    logit = tf.nn.xw_plus_b(x=output, weights=self.fc_W, biases=self.fc_b)        
                    # final_state += tf.transpose(tf.transpose(tf.concat(state, axis=1))) # * self.mask[:, i])
                    # final_state += output
                    prob.append(tf.nn.sigmoid(logit))

            # logit = tf.nn.xw_plus_b(x=final_state, weights=self.fc_W, biases=self.fc_b)
            prob =tf.transpose(tf.squeeze(tf.stack(prob)))
        # for v in tf.all_variables():
        #     if v.name.startswith(vs_lstm.name):
        #         self.theta_G.append(v)
        self.reuse = True
        return prob

    def build_disc(self):
        fake = self.build(self.sentence)
        real = self.build(self.truth)
        self.real = real
        self.fake = fake
        # self.reward = tf.reduce_mean(tf.log(tf.clip_by_value(fake, 1e-16, 1.0)), axis=0)
        self.reward = tf.reduce_mean(tf.log(tf.clip_by_value(fake[:, -1], 1e-16, 1.0)), axis=0)
        self.d_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(real, 1e-16, 1.0))
             + tf.log(tf.clip_by_value(1. - fake, 1e-16, 1.0)))
        # self.reward = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logit, labels = self.labels))
        # self.d_loss = loss_real + loss_fake
        self.train_op = tf.train.AdamOptimizer().minimize(self.d_loss)
