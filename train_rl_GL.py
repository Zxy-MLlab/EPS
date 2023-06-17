import re
import os
import json
import numpy as np
import copy
import tqdm
import random


IN_PATH = 'rl_data'
OUT_PATH = 'rl_data'
SAMPLES = 3
H_T_SAMPLES = 3
EPSILON = 0.05
GAMMA = 0.9

class Reader():
    def __init__(self, in_path, out_path):
        self.in_path = in_path
        self.out_path = out_path

        return

    def read_origin_data(self, prefix):
        file_path = os.path.join(self.in_path, prefix + '.json')
        origin_data = json.load(open(file_path))

        return origin_data

    def read_relations(self):
        rel2id = json.load(open(os.path.join(self.out_path, 'rel2id.json'), 'r'))
        id2rel = {v:u for u,v in rel2id.items()}

        return rel2id, id2rel

    def read_word_id(self):
        word2id = json.load(open(os.path.join(self.out_path, 'word2id.json')))
        ner2id = json.load(open(os.path.join(self.out_path, 'ner2id.json')))

        char2id = json.load(open(os.path.join(self.out_path, 'char2id.json')))

        return word2id, ner2id, char2id

    def read_vec(self):
        word_vec = np.load(os.path.join(self.out_path, 'vec.npy'))
        char_vec = np.load(os.path.join(self.out_path, 'char_vec.npy'))
        croef_vec = np.load(os.path.join(self.out_path, 'coref_vec.npy'))
        ner_vec = np.load(os.path.join(self.out_path, 'ner_vec.npy'))
        dis_vec = np.load(os.path.join(self.out_path, 'dis_vec.npy'))

        return word_vec, char_vec, croef_vec, ner_vec, dis_vec

def train_rl(processer, CModel, PModel, MODEL_PATH, TRAIN_EPOCHES):
    reader = Reader(in_path=IN_PATH, out_path=OUT_PATH)

    prefix = 'dev_train'
    origin_data = reader.read_origin_data(prefix)
    # random.shuffle(origin_data)
    train_data = origin_data[800:900]

    train_rewards = []
    best_reward = float('-inf')
    for epoch in range(TRAIN_EPOCHES):
        e_rewards = []

        for i, _ in enumerate(tqdm.tqdm(train_data)):
            data = copy.deepcopy(train_data[i])
            vertexSet = copy.deepcopy(data['vertexSet'])
            labels = copy.deepcopy(data['labels'])
            na_triples = copy.deepcopy(data['na_triple'])
            all_triples = copy.deepcopy(data['all_triples'])
            all_sents = list(range(len(data['sents'])))
            ins = copy.deepcopy(train_data[i])

            input_data = processer.process_cldata(ins=data)
            logits,_ = CModel.get_rewards(input_data)

            select_order = []
            for x in range(len(all_triples)):
                if (all_triples[x]['e'] == False) & (np.max(logits[0][x]) < 0.85):
                    select_order.append(x)

            if len(select_order) == 0:
                continue
            random.shuffle(select_order)

            # select_order = select_order[: min(len(select_order), H_T_SAMPLES)]

            s_context_idxs = []
            s_context_pos = []
            s_context_ner = []
            s_h_mapping = []
            s_t_mapping = []
            s_sen_idxs = []
            s_sen_pos = []
            s_sen_ner = []
            s_actions = []
            s_rewards = []

            for s_a in range(SAMPLES):
                t_context_idxs = []
                t_context_pos = []
                t_context_ner = []
                t_h_mapping = []
                t_t_mapping = []
                t_sen_idxs = []
                t_sen_pos = []
                t_sen_ner = []
                t_actions = []
                t_sen_len = []

                for s_o in select_order:
                    l_context_idxs = []
                    l_context_pos = []
                    l_context_ner = []
                    l_h_mapping = []
                    l_t_mapping = []
                    l_sen_idxs = []
                    l_sen_pos = []
                    l_sen_ner = []
                    l_actions = []

                    label = copy.deepcopy(all_triples[s_o])
                    rl_evidence_sents = label['evidence']

                    for s_i in all_sents:
                        if s_i in rl_evidence_sents:
                            continue

                        context_idxs, context_pos, context_ner, \
                        h_mapping, t_mapping = processer.process_his_data(data, label, rl_evidence_sents)

                        sen_idx, sen_pos, sen_ner = processer.process_cur_data(data, s_i)

                        _, prob = PModel.get_action(context_idxs=context_idxs,
                                                    context_pos=context_pos,
                                                    context_ner=context_ner,
                                                    h_mapping=h_mapping, t_mapping=t_mapping,
                                                    sen_idx=sen_idx, sen_pos=sen_pos,
                                                    sen_ner=sen_ner)

                        if random.random() > EPSILON:
                            action = (0 if random.random() < prob[0][0] else 1)
                        else:
                            action = (1 if random.random() < prob[0][0] else 0)

                        if action == 1:
                            rl_evidence_sents.append(s_i)

                        # store his data
                        l_context_idxs.append(context_idxs.copy().tolist()[0])
                        l_context_pos.append(context_pos.copy().tolist()[0])
                        l_context_ner.append(context_ner.copy().tolist()[0])
                        l_h_mapping.append(h_mapping.copy().tolist()[0][0])
                        l_t_mapping.append(t_mapping.copy().tolist()[0][0])
                        l_sen_idxs.append(sen_idx.copy().tolist()[0])
                        l_sen_pos.append(sen_pos.copy().tolist()[0])
                        l_sen_ner.append(sen_ner.copy().tolist()[0])
                        l_actions.append(action)

                    rl_evidence_sents.sort()
                    if s_o < len(labels):
                        ins['labels'][s_o]['evidence'] = rl_evidence_sents
                    else:
                        ins['na_triple'][s_o-len(labels)]['evidence'] = rl_evidence_sents

                    t_context_idxs.append(l_context_idxs)
                    t_context_pos.append(l_context_pos)
                    t_context_ner.append(l_context_ner)
                    t_h_mapping.append(l_h_mapping)
                    t_t_mapping.append(l_h_mapping)
                    t_sen_idxs.append(l_sen_idxs)
                    t_sen_pos.append(l_sen_pos)
                    t_sen_ner.append(l_sen_ner)
                    t_actions.append(l_actions)
                    t_sen_len.append(len(rl_evidence_sents))

                input_data = processer.process_cldata(
                    ins)

                _, all_rewards = CModel.get_rewards(input_data)

                # t_rewards = [all_rewards[x] + ((float(len(all_sents)-t_sen_len[x_idx]) / len(all_sents)) **2 *0.15) for x_idx, x in enumerate(select_order)]
                t_rewards = [all_rewards[x] for x in select_order]

                s_context_idxs.append(t_context_idxs)
                s_context_pos.append(t_context_pos)
                s_context_ner.append(t_context_ner)
                s_h_mapping.append(t_h_mapping)
                s_t_mapping.append(t_t_mapping)
                s_sen_idxs.append(t_sen_idxs)
                s_sen_pos.append(t_sen_pos)
                s_sen_ner.append(t_sen_ner)
                s_actions.append(t_actions)
                s_rewards.append(t_rewards)

            s_rewards = np.array(s_rewards)
            ave_rewards = np.mean(s_rewards, axis=0)

            e_rewards.extend(ave_rewards)

            for s_o in range(len(select_order)):
                ave_reward = ave_rewards[s_o]
                for s_a in range(SAMPLES):
                    context_idxs = np.array(s_context_idxs[s_a][s_o])
                    context_pos = np.array(s_context_pos[s_a][s_o])
                    context_ner = np.array(s_context_ner[s_a][s_o])
                    h_mapping = np.expand_dims(np.array(s_h_mapping[s_a][s_o]), axis=1)
                    t_mapping = np.expand_dims(np.array(s_t_mapping[s_a][s_o]), axis=1)
                    sen_idxs = np.array(s_sen_idxs[s_a][s_o])
                    sen_pos = np.array(s_sen_pos[s_a][s_o])
                    sen_ner = np.array(s_sen_ner[s_a][s_o])
                    actions = np.array(s_actions[s_a][s_o])
                    reward = s_rewards[s_a][s_o] - ave_reward

                    if len(context_idxs) == 0:
                        continue

                    rewards = [0.0 for x in range(len(actions))]
                    rewards[len(rewards) - 1] = reward

                    values = [0.0 for x in range(len(actions))]
                    running_add = 0
                    for r_i in reversed(range(0, len(rewards))):
                        running_add = running_add * GAMMA + rewards[r_i]
                        values[r_i] = running_add
                    values = np.array(values)

                    PModel.train(context_idxs=context_idxs, context_pos=context_pos, context_ner=context_ner,
                                 h_mapping=h_mapping, t_mapping=t_mapping,
                                 sen_idxs=sen_idxs, sen_pos=sen_pos, sen_ner=sen_ner,
                                 actions=actions, values=values)

        epoch_mean_reward = np.mean(np.array(e_rewards))
        print("train epoch: %s; reward: %s"%(str(epoch), str(epoch_mean_reward)))
        train_rewards.append(epoch_mean_reward)

        if epoch_mean_reward > best_reward:
            model_path = MODEL_PATH + '/' + 'PolicyModel/best/pmodel.ckpt'
            PModel.save(model_path)
            best_reward = epoch_mean_reward

    model_path = MODEL_PATH + '/' + 'PolicyModel/pmodel.ckpt'
    PModel.save(model_path)

    epoches = [x+1 for x in range(len(train_rewards))]
    reward_path = MODEL_PATH + '/' + 'rewards.csv'
    with open(reward_path, 'a', encoding='utf-8') as f_write:
        for x,reward in zip(epoches, train_rewards):
            f_write.write(str(x))
            f_write.write('\t')
            f_write.write(str(reward))
            f_write.write('\n')
    f_write.close()

    return