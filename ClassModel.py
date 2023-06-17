import re
import os
import numpy as np
import tensorflow as tf
from code_update_3.cr_model_train.models import BiLSTM_GL

class ClassModel():
    def __init__(self, sess, model_path, reader, MAX_LENGTH, CLASS_NUM, model_flag):
        self.sess = sess
        self.model_path = model_path
        self.reader = reader

        word_vec, char_vec, coref_vec, ner_vec, dis_vec = reader.read_vec()
        self.Model = BiLSTM_GL.BiLSTM(MAX_LENGTH, CLASS_NUM, word_vec, coref_vec, ner_vec, dis_vec)

        self.saver = tf.train.Saver()

        if model_flag:
            self.saver.restore(self.sess, os.path.join(self.model_path, 'ClassModel/cmodel.ckpt'))
        else:
            init = tf.global_variables_initializer()
            self.sess.run(init)

        return

    def train(self, data, lr):
        global_context_idxs = data['global_context_idxs']
        global_context_pos = data['global_context_pos']
        global_context_ner = data['global_context_ner']
        global_h_mapping = data['global_h_mapping']
        global_t_mapping = data['global_t_mapping']
        global_ht_pos = data['global_ht_pos']

        local_context_idxs = data['local_context_idxs']
        local_context_pos = data['local_context_pos']
        local_context_ner = data['local_context_ner']
        local_h_mapping = np.expand_dims(data['local_h_mapping'], axis=2)
        local_t_mapping = np.expand_dims(data['local_t_mapping'], axis=2)
        local_ht_pos = data['local_ht_pos']

        relation_multi_label = data['relation_multi_label']
        relation_mask = np.expand_dims(data['relation_mask'], axis=2)
        max_global_wn = data['max_global_wn']
        max_local_wn = data['max_local_wn']
        max_h_t_cnt = data['max_h_t_cnt']

        global_dis_h_2_t = global_ht_pos + 10
        global_dis_t_2_h = -global_ht_pos + 10

        local_dis_h_2_t = local_ht_pos + 10
        local_dis_t_2_h = -local_ht_pos + 10

        pre, loss, _ = self.sess.run([self.Model.pre, self.Model.loss, self.Model.train_op], feed_dict={
            self.Model.global_context_idx: global_context_idxs,
            self.Model.global_pos: global_context_pos,
            self.Model.global_context_ner: global_context_ner,
            self.Model.global_h_mapping: global_h_mapping,
            self.Model.global_t_mapping: global_t_mapping,
            self.Model.global_dis_h_2_t: global_dis_h_2_t,
            self.Model.global_dis_t_2_h: global_dis_t_2_h,
            self.Model.local_context_idxs: local_context_idxs,
            self.Model.local_pos: local_context_pos,
            self.Model.local_context_ner: local_context_ner,
            self.Model.local_h_mapping: local_h_mapping,
            self.Model.local_t_mapping: local_t_mapping,
            self.Model.local_dis_h_2_t: local_dis_h_2_t,
            self.Model.local_dis_t_2_h: local_dis_t_2_h,
            self.Model.relation_multi_label: relation_multi_label,
            self.Model.relation_mask: relation_mask,
            self.Model.h_t_limit: max_h_t_cnt,
            self.Model.global_word_num: max_global_wn,
            self.Model.local_word_num: max_local_wn,
            self.Model.lr: lr,
            self.Model.keep_prob: 0.5,
        })

        return pre, loss

    def test(self, data):
        global_context_idxs = data['global_context_idxs']
        global_context_pos = data['global_context_pos']
        global_context_ner = data['global_context_ner']
        global_h_mapping = data['global_h_mapping']
        global_t_mapping = data['global_t_mapping']
        global_ht_pos = data['global_ht_pos']

        local_context_idxs = data['local_context_idxs']
        local_context_pos = data['local_context_pos']
        local_context_ner = data['local_context_ner']
        local_h_mapping = np.expand_dims(data['local_h_mapping'], axis=2)
        local_t_mapping = np.expand_dims(data['local_t_mapping'], axis=2)
        local_ht_pos = data['local_ht_pos']

        relation_multi_label = data['relation_multi_label']
        relation_mask = np.expand_dims(data['relation_mask'], axis=2)

        max_global_wn = data['max_global_wn']
        max_local_wn = data['max_local_wn']
        max_h_t_cnt = data['max_h_t_cnt']

        global_dis_h_2_t = global_ht_pos + 10
        global_dis_t_2_h = -global_ht_pos + 10

        local_dis_h_2_t = local_ht_pos + 10
        local_dis_t_2_h = -local_ht_pos + 10

        pre, loss = self.sess.run([self.Model.dev_pre, self.Model.loss], feed_dict={
            self.Model.global_context_idx: global_context_idxs,
            self.Model.global_pos: global_context_pos,
            self.Model.global_context_ner: global_context_ner,
            self.Model.global_h_mapping: global_h_mapping,
            self.Model.global_t_mapping: global_t_mapping,
            self.Model.global_dis_h_2_t: global_dis_h_2_t,
            self.Model.global_dis_t_2_h: global_dis_t_2_h,
            self.Model.local_context_idxs: local_context_idxs,
            self.Model.local_pos: local_context_pos,
            self.Model.local_context_ner: local_context_ner,
            self.Model.local_h_mapping: local_h_mapping,
            self.Model.local_t_mapping: local_t_mapping,
            self.Model.local_dis_h_2_t: local_dis_h_2_t,
            self.Model.local_dis_t_2_h: local_dis_t_2_h,
            self.Model.relation_multi_label: relation_multi_label,
            self.Model.relation_mask: relation_mask,
            self.Model.h_t_limit: max_h_t_cnt,
            self.Model.global_word_num: max_global_wn,
            self.Model.local_word_num: max_local_wn,
            self.Model.keep_prob: 1.0,
        })

        return pre, loss

    def get_rewards(self, data):
        global_context_idxs = data['global_context_idxs']
        global_context_pos = data['global_context_pos']
        global_context_ner = data['global_context_ner']
        global_h_mapping = data['global_h_mapping']
        global_t_mapping = data['global_t_mapping']
        global_ht_pos = data['global_ht_pos']
        local_context_idxs = data['local_context_idxs']
        local_context_pos = data['local_context_pos']
        local_context_ner = data['local_context_ner']
        local_h_mapping = np.expand_dims(data['local_h_mapping'], axis=2)
        local_t_mapping = np.expand_dims(data['local_t_mapping'], axis=2)
        local_ht_pos = data['local_ht_pos']
        relation_multi_label = data['relation_multi_label']
        max_global_wn = data['max_global_wn']
        max_local_wn = data['max_local_wn']
        max_h_t_cnt = data['max_h_t_cnt']

        global_dis_h_2_t = global_ht_pos + 10
        global_dis_t_2_h = -global_ht_pos + 10

        local_dis_h_2_t = local_ht_pos + 10
        local_dis_t_2_h = -local_ht_pos + 10

        output, logits, log1, log2, rewards = self.sess.run(
            [self.Model.output, self.Model.logits, self.Model.log1, self.Model.log2, self.Model.rewards], feed_dict={
                self.Model.global_context_idx: global_context_idxs,
                self.Model.global_pos: global_context_pos,
                self.Model.global_context_ner: global_context_ner,
                self.Model.global_h_mapping: global_h_mapping,
                self.Model.global_t_mapping: global_t_mapping,
                self.Model.global_dis_h_2_t: global_dis_h_2_t,
                self.Model.global_dis_t_2_h: global_dis_t_2_h,
                self.Model.local_context_idxs: local_context_idxs,
                self.Model.local_pos: local_context_pos,
                self.Model.local_context_ner: local_context_ner,
                self.Model.local_h_mapping: local_h_mapping,
                self.Model.local_t_mapping: local_t_mapping,
                self.Model.local_dis_h_2_t: local_dis_h_2_t,
                self.Model.local_dis_t_2_h: local_dis_t_2_h,
                self.Model.relation_multi_label: relation_multi_label,
                self.Model.global_word_num: max_global_wn,
                self.Model.local_word_num: max_local_wn,
                self.Model.h_t_limit: max_h_t_cnt,
                self.Model.keep_prob: 1.0,
            })

        rewards = rewards[0]

        return logits, rewards

    def restore(self, model_path):
        self.saver.restore(self.sess, model_path)

        return

    def save(self, model_path):
        self.saver.save(self.sess, model_path)

        return