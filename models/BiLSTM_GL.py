import re
import tensorflow as tf
import tensorflow.contrib as contrib
import tensorflow.contrib.layers as layers

class BiLSTM():
    def __init__(self, max_length, relation_num, word_vec, coref_vec, ner_vec, dis_vec):
        self.max_length = max_length  # max context length
        self.relation_num = relation_num  # relation class num

        self.word_emb = tf.get_variable(initializer=word_vec, name='word_embedding', trainable=False)  # word embedding
        self.coref_emb = tf.get_variable(initializer=coref_vec, name='coref_embedding',
                                         trainable=False)  # coreference embedding
        self.ner_emb = tf.get_variable(initializer=ner_vec, name='ner_embedding', trainable=False)  # type embedding
        self.dis_emb = tf.get_variable(initializer=dis_vec, name='dis_embedding', trainable=False)  # dis embedding

        self.global_context_idx = tf.placeholder(tf.int32, shape=[None, None], name='global_context_idxs')
        self.global_pos = tf.placeholder(tf.int32, shape=[None, None], name='global_pos')
        self.global_context_ner = tf.placeholder(tf.int32, shape=[None, None], name='global_context_ner')
        self.global_h_mapping = tf.placeholder(tf.float32, shape=[None, None, None], name='global_h_mapping')
        self.global_t_mapping = tf.placeholder(tf.float32, shape=[None, None, None], name='global_t_mapping')
        self.global_dis_h_2_t = tf.placeholder(tf.int32, shape=[None, None], name='global_dis_h_2_t')
        self.global_dis_t_2_h = tf.placeholder(tf.int32, shape=[None, None], name='global_dis_t_2_h')

        self.local_context_idxs = tf.placeholder(tf.int32, shape=[None, None, None],
                                           name='local_context_idxs')  # context word ids
        self.local_pos = tf.placeholder(tf.int32, shape=[None, None, None], name='local_pos')  # entity mention pos
        self.local_context_ner = tf.placeholder(tf.int32, shape=[None, None, None],
                                          name='local_context_ner')  # context word ners
        self.local_h_mapping = tf.placeholder(tf.float32, shape=[None, None, 1, None], name='local_h_mapping')  # h mapping
        self.local_t_mapping = tf.placeholder(tf.float32, shape=[None, None, 1, None], name='local_t_mapping')  # t_mapping
        self.local_dis_h_2_t = tf.placeholder(tf.int64, shape=[None, None],
                                        name='local_dis_h_2_t')  # distance heart to termination entity
        self.local_dis_t_2_h = tf.placeholder(tf.int32, shape=[None, None],
                                        name='local_dis_t_2_h')  # distance termination to heart entity
        self.relation_multi_label = tf.placeholder(tf.float32, shape=[None, None, self.relation_num],
                                                   name='relation_multi_label')  # relation multi class
        self.relation_mask = tf.placeholder(tf.float32, shape=[None, None, 1], name='relation_mask')  # relation mask

        self.global_word_num = tf.placeholder(tf.int32)
        self.local_word_num = tf.placeholder(tf.int32)  # heart and termination relation num
        self.h_t_limit = tf.placeholder(tf.int32)
        self.keep_prob = tf.placeholder(tf.float32)  # dropout keep prob
        self.lr = tf.placeholder(tf.float32) # learning rate

        self.hidden_size = 128
        self.layer_num = 1
        self.batch_size = 32

        self.activation = tf.nn.relu

        global_bilstm_out = self.global_bilstm(self.global_context_idx, self.global_pos, self.global_context_ner)
        global_context = self.linear(global_bilstm_out, self.global_word_num)

        global_start_output = tf.matmul(self.global_h_mapping, global_context)
        global_end_output = tf.matmul(self.global_t_mapping, global_context)

        global_dis_h2t_ebd = tf.nn.embedding_lookup(self.dis_emb, self.global_dis_h_2_t)
        global_dis_t2h_ebd = tf.nn.embedding_lookup(self.dis_emb, self.global_dis_t_2_h)

        global_start_rep = tf.concat([global_start_output, global_dis_h2t_ebd], axis=-1)
        global_end_rep = tf.concat([global_end_output, global_dis_t2h_ebd], axis=-1)

        local_bilstm_out = self.local_bilstm(self.local_context_idxs, self.local_pos, self.local_context_ner)
        local_context = self.linear(local_bilstm_out, self.local_word_num)
        local_context = tf.reshape(local_context, shape=[-1, self.h_t_limit, self.local_word_num, local_context.shape[-1]])

        local_start_output = tf.matmul(self.local_h_mapping, local_context)
        local_end_output = tf.matmul(self.local_t_mapping, local_context)

        local_start_output = tf.squeeze(local_start_output, axis=2)
        local_end_output = tf.squeeze(local_end_output, axis=2)

        local_dis_h2t_ebd = tf.nn.embedding_lookup(self.dis_emb, self.local_dis_h_2_t)
        local_dis_t2h_ebd = tf.nn.embedding_lookup(self.dis_emb, self.local_dis_t_2_h)

        local_start_rep = tf.concat([local_start_output, local_dis_h2t_ebd], axis=-1)
        local_end_rep = tf.concat([local_end_output, local_dis_t2h_ebd], axis=-1)

        start_rep = tf.concat([global_start_rep, local_start_rep], axis=-1)
        end_rep = tf.concat([global_end_rep, local_end_rep], axis=-1)

        output = self.bilinear(start_rep, end_rep, self.relation_num)

        self.dev_pre,self.pre = self.pred(output)

        self.loss = self.com_loss(self.relation_multi_label, output)
        self.rewards = self.get_rewards(self.relation_multi_label, output)

        self.train_op = self.optimizer(self.lr, self.loss)

        return

    def global_bilstm(self, context_ids, pos, context_ner):
        context_ebd = tf.nn.embedding_lookup(self.word_emb, context_ids)
        pos_ebd = tf.nn.embedding_lookup(self.coref_emb, pos)
        context_ner_ebd = tf.nn.embedding_lookup(self.ner_emb, context_ner)

        sent = tf.concat([context_ebd, pos_ebd, context_ner_ebd], axis=-1)

        lstm_cell_fw = contrib.rnn.LSTMCell(self.hidden_size, name='fw_global')
        lstm_cell_bw = contrib.rnn.LSTMCell(self.hidden_size, name='bw_global')

        output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, sent, dtype=tf.float32)

        output = tf.concat(output, axis=-1)

        return output

    def local_bilstm(self, context_ids, pos, context_ner):
        context_ebd = tf.nn.embedding_lookup(self.word_emb, context_ids)
        pos_ebd = tf.nn.embedding_lookup(self.coref_emb, pos)
        context_ner_ebd = tf.nn.embedding_lookup(self.ner_emb, context_ner)

        sent = tf.concat([context_ebd, pos_ebd, context_ner_ebd], axis=-1)
        sent = tf.reshape(sent, shape=[-1, self.local_word_num, sent.shape[-1]])

        lstm_cell_fw = contrib.rnn.LSTMCell(self.hidden_size)
        lstm_cell_bw = contrib.rnn.LSTMCell(self.hidden_size)

        output,_ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, sent, dtype=tf.float32)

        output = tf.concat(output, axis=-1)

        return output

    def linear(self, input, dim):

        input = tf.reshape(input, shape=[-1, input.shape[2]])

        out = layers.fully_connected(inputs=input, num_outputs=2 * self.hidden_size, activation_fn=self.activation)

        out = tf.nn.dropout(out, keep_prob=self.keep_prob)

        out = tf.reshape(out, shape=[-1, dim, out.shape[-1]])

        return out

    def bilinear(self, input1, input2, out_size):
        bi_input1 = tf.reshape(input1, [-1, input1.shape[2]])
        bi_input2 = tf.reshape(input2, [-1, input2.shape[2]])
        outs = []

        scale = float(1.0 / 276) ** 0.5

        for i in range(out_size):
            w_name = 'w' + str(i)
            w = self.weight_variable(name=w_name, shape=[bi_input1.shape[1], bi_input2.shape[1]], scale=scale)
            out = tf.matmul(bi_input1, w)
            out = tf.multiply(out, bi_input2)
            out = tf.reduce_sum(out, axis=1)
            out = tf.reshape(out, shape=[-1, 1])
            outs.append(out)

        output = tf.concat(outs, axis=1)
        b = self.bias_variable(name='b', shape=[out_size], scale=scale)
        output = output + b

        output = tf.reshape(output, shape=[-1, self.h_t_limit, output.shape[1]])

        return output

    def pred(self, input):
        dev_pre = tf.nn.sigmoid(input)
        pre = tf.argmax(dev_pre, axis=-1)

        return dev_pre, pre

    def com_loss(self, labels, logits):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_sum(loss * self.relation_mask) / (self.relation_num * tf.reduce_sum(self.relation_mask))

        return loss

    def get_rewards(self, labels, logits):
        self.output = logits
        logits = tf.nn.sigmoid(logits)
        logits = tf.clip_by_value(logits, 1e-3, 1-(1e-3))
        self.logits = logits

        log1 = -labels * tf.log(1-logits)
        log2 = -(1-labels) * tf.log(logits)

        log = log1 + log2

        self.log1 = log1
        self.log2 = log2
        rewards = tf.reduce_mean(log, axis=2)

        return rewards

    def optimizer(self, lr, loss):
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)

        return train_op

    def weight_variable(self, name, shape, scale):
        initial = tf.random_uniform_initializer(minval=-scale, maxval=scale)
        return tf.get_variable(name=name, shape=shape, initializer=initial)

    def bias_variable(self, name, shape, scale):
        initial = tf.random_uniform_initializer(minval=-scale, maxval=scale)
        return tf.get_variable(name=name, shape=shape, initializer=initial)