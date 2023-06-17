import re
import tensorflow as tf
from models import MC

class PolicyModel():
    def __init__(self, sess, reader, MAX_LENGTH, RL_LEARNING_RATE):
        self.sess = sess
        self.reader = reader

        word_vec, char_ver, coref_vec, ner_vec, dis_vec = reader.read_vec()

        self.agent = MC.Agent(MAX_LENGTH, RL_LEARNING_RATE, word_vec, coref_vec, ner_vec)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.saver = tf.train.Saver()

        return

    def get_action(self, context_idxs, context_pos, context_ner, h_mapping, t_mapping, sen_idx, sen_pos, sen_ner):
        max_word_nums = context_idxs.shape[1]

        his_state, cur_state, state, action, prob = self.sess.run(
            [self.agent.his_state, self.agent.cur_state, self.agent.state, self.agent.a, self.agent.prob], feed_dict={
                self.agent.context_idx: context_idxs,
                self.agent.context_pos: context_pos,
                self.agent.context_ner: context_ner,
                self.agent.h_mapping: h_mapping,
                self.agent.t_mapping: t_mapping,
                self.agent.sen_idx: sen_idx,
                self.agent.sen_pos: sen_pos,
                self.agent.sen_ner: sen_ner,
                self.agent.word_nums: max_word_nums,
                self.agent.keep_prob: 1.0,
            })

        action = action[0]

        return action, prob

    def train(self, context_idxs, context_pos, context_ner, h_mapping, t_mapping,
              sen_idxs, sen_pos, sen_ner, actions, values):
        loss, _ = self.sess.run([self.agent.loss, self.agent.train_op], feed_dict={
            self.agent.context_idx: context_idxs,
            self.agent.context_pos: context_pos,
            self.agent.context_ner: context_ner,
            self.agent.h_mapping: h_mapping,
            self.agent.t_mapping: t_mapping,
            self.agent.sen_idx: sen_idxs,
            self.agent.sen_pos: sen_pos,
            self.agent.sen_ner: sen_ner,
            self.agent.s_action: actions,
            self.agent.s_value: values,
            self.agent.keep_prob: 0.6,
        })

        return

    def save(self, model_path):
        self.saver.save(self.sess, model_path)

        return

    def restore(self, model_path):
        self.saver.restore(self.sess, model_path)

        return