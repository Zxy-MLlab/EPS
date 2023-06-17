import re
import tensorflow as tf
import tensorflow.contrib as contrib
import tensorflow.contrib.layers as layers

class Agent():
    def __init__(self, max_length, lr, word_vec, coref_vec, ner_vec):
        self.max_length = max_length
        self.lr = lr

        self.word_emb = tf.get_variable(initializer=word_vec, name='rl_word_ebd', trainable=False)
        self.coref_emb = tf.get_variable(initializer=coref_vec, name='rl_coref_ebd', trainable=False)
        self.ner_emb = tf.get_variable(initializer=ner_vec, name='rl_ner_ebd', trainable=False)

        self.context_idx = tf.placeholder(tf.int64, shape=[None, None], name='context_idx')
        self.context_pos = tf.placeholder(tf.int64, shape=[None, None], name='context_pos')
        self.context_ner = tf.placeholder(tf.int64, shape=[None, None], name='context_ner')

        self.sen_idx = tf.placeholder(tf.int64, shape=[None, None], name='sen_idx')
        self.sen_pos = tf.placeholder(tf.int64, shape=[None, None], name='sen_pos')
        self.sen_ner = tf.placeholder(tf.int64, shape=[None, None], name='sen_ner')

        self.h_mapping = tf.placeholder(tf.float32, shape=[None, 1, None], name='h_mapping')
        self.t_mapping = tf.placeholder(tf.float32, shape=[None, 1, None], name='t_mapping')

        self.s_action = tf.placeholder(tf.int64, shape=[None, ], name='s_action')
        self.s_value = tf.placeholder(tf.float32, shape=[None, ], name='s_value')
        self.word_nums = tf.placeholder(tf.int64)
        self.keep_prob = tf.placeholder(tf.float32)

        self.out_channels = 5
        self.kernel_size = 5
        self.stride = 1
        self.out_size = 200
        self.hidden_size = 128
        self.action_space = 2
        self.padding = 'same'

        self.activation = tf.nn.tanh

        context_out, _ = self.bilstm(self.context_idx, self.context_pos, self.context_ner, 'context')
        context_out = self.linear(context_out)

        start_re_output = tf.matmul(self.h_mapping, context_out)
        end_re_output = tf.matmul(self.t_mapping, context_out)

        self.his_state = tf.reduce_mean([start_re_output, end_re_output], axis=0, keepdims=False)
        self.his_state = tf.squeeze(self.his_state, axis=1)

        _, sen_out = self.bilstm(self.sen_idx, self.sen_pos, self.sen_ner, 'sen')
        sen_out = self.linear(sen_out)

        self.cur_state = sen_out

        self.state = tf.reduce_mean([self.his_state, self.cur_state], axis=0, keepdims=False)

        self.a, self.prob, self.log = self.action(self.state)

        self.neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.log, labels=self.s_action)
        self.loss = tf.reduce_mean(self.neg_log_prob * self.s_value)

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        return

    def bilstm(self, context_idxs, pos, context_ner, name):
        context_ebd = tf.nn.embedding_lookup(self.word_emb, context_idxs)
        pos_ebd = tf.nn.embedding_lookup(self.coref_emb, pos)
        context_ner_ebd = tf.nn.embedding_lookup(self.ner_emb, context_ner)

        sent = tf.concat([context_ebd, pos_ebd, context_ner_ebd], axis=-1)

        fw_name = 'fw_' + name
        bw_name = 'bw_' + name

        lstm_cell_fw = contrib.rnn.LSTMCell(self.hidden_size, name=fw_name)
        lstm_cell_bw = contrib.rnn.LSTMCell(self.hidden_size, name=bw_name)

        output, hidden_output = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, sent, dtype=tf.float32)

        output = output[1]
        hidden_output = hidden_output[1][1]

        return output, hidden_output

    def linear(self, input):
        out = layers.fully_connected(inputs=input, num_outputs=2 * self.hidden_size, activation_fn=self.activation)

        out = tf.nn.dropout(out, keep_prob=self.keep_prob)

        return out

    def action(self, state):
        h_layer = layers.fully_connected(inputs=state, num_outputs=2 * self.out_size, activation_fn=self.activation)

        s_layer = layers.fully_connected(inputs=h_layer, num_outputs=self.action_space, activation_fn=None)

        a = tf.argmax(tf.nn.softmax(s_layer), axis=1)
        prob = tf.nn.softmax(s_layer)

        return a, prob, s_layer

    def bilinear(self, input1, input2, out_size):
        outs = []
        scale = float(1.0 / 256) ** 0.5

        for i in range(out_size):
            w_name = 'w' + str(i)
            w = self.weight_variable(name=w_name, shape=[input1.shape[1], input2.shape[1]], scale=scale)
            out = tf.matmul(input1, w)
            out = tf.multiply(out, input2)
            out = tf.reduce_sum(out, axis=1)
            out = tf.reshape(out, shape=[-1, 1])
            outs.append(out)

        output = tf.concat(outs, axis=1)
        b = self.bias_variable(name='b', shape=[out_size], scale=scale)
        output = output + b

        return output

    def weight_variable(self, name, shape, scale):
        initial = tf.random_uniform_initializer(minval=-scale, maxval=scale)
        return tf.get_variable(name=name, shape=shape, initializer=initial)

    def bias_variable(self, name, shape, scale):
        initial = tf.random_uniform_initializer(minval=-scale, maxval=scale)
        return tf.get_variable(name=name, shape=shape, initializer=initial)