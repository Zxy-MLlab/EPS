import re
import numpy as np
import os
import json
import random
import sklearn.metrics
import copy

H_T_LIMIT = 1800
MAX_LENGTH = 512
CLASS_NUM = 97
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
PRINT_LOSS = True

dis2idx = np.zeros((MAX_LENGTH), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9

class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0
    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1
    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total
    def clear(self):
        self.correct = 0
        self.total = 0

def read_train_data(DATA_PATH='prepro_data'):
    prefix = 'dev_train'

    train_file = json.load(open(os.path.join(DATA_PATH, prefix + '.json')))

    prefix = 'dev_dev'

    test_file = json.load(open(os.path.join(DATA_PATH, prefix + '.json')))

    return train_file, test_file

def process_train_data(train_file, word2id, ner2id):
    print("start process train data...")
    global dis2idx

    train_data_size = len(train_file)
    train_order = list(range(train_data_size))

    random.shuffle(train_order)
    batch_num = train_data_size // BATCH_SIZE
    if train_data_size % BATCH_SIZE != 0:
        batch_num = batch_num + 1

    for i in range(batch_num):
        start_index = i * BATCH_SIZE
        cur_index = min(BATCH_SIZE, train_data_size-start_index)
        cur_order = list(train_order[start_index: start_index+cur_index])

        global_context_idxs = np.zeros((cur_index, MAX_LENGTH), dtype=np.int32)
        global_context_pos = np.zeros((cur_index, MAX_LENGTH), dtype=np.int32)
        global_context_ner = np.zeros((cur_index, MAX_LENGTH), dtype=np.int32)
        global_h_mapping = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.float32)
        global_t_mapping = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.float32)
        global_ht_pos = np.zeros((cur_index, H_T_LIMIT), dtype=np.int32)

        local_context_idxs = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.int32)
        local_context_pos = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.int32)
        local_context_ner = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.int32)
        local_h_mapping = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.float32)
        local_t_mapping = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.float32)
        local_ht_pos = np.zeros((cur_index, H_T_LIMIT), dtype=np.int32)

        relation_multi_label = np.zeros((cur_index, H_T_LIMIT, CLASS_NUM), dtype=np.float32)
        relation_label = np.full([cur_index, H_T_LIMIT], fill_value=-100)
        relation_mask = np.zeros((cur_index, H_T_LIMIT), dtype=np.float32)

        max_global_wn = 0
        max_local_wn = 0
        max_h_t_cnt = 0

        for k, index in enumerate(cur_order):
            ins = train_file[index]

            words = []
            for sent in ins['sents']:
                words += sent

            max_global_wn = max(len(words), max_global_wn)
            for t, word in enumerate(words):
                word = word.lower()
                if t < MAX_LENGTH:
                    if word in word2id:
                        global_context_idxs[k, t] = word2id[word]
                    else:
                        global_context_idxs[k, t] = word2id['UNK']

            for t in range(t + 1, MAX_LENGTH):
                global_context_idxs[k, t] = word2id['BLANK']

            vertexSet = copy.deepcopy(ins['vertexSet'])
            for idx, vertex in enumerate(vertexSet, 1):
                for v in vertex:
                    global_context_pos[k, v['pos'][0]:v['pos'][1]] = idx
                    global_context_ner[k, v['pos'][0]:v['pos'][1]] = ner2id[v['type']]

            h_t_cnt = 0

            all_triples = copy.deepcopy(ins['all_triples'])

            random.shuffle(all_triples)

            triple_dict = dict()

            for j, triple in enumerate(all_triples):
                h = triple['h']
                t = triple['t']
                r = triple['r']

                if (h,t) not in triple_dict.keys():
                    triple_dict[(h,t)] = j
                    relation_multi_label[k, j, r] = 1
                    max_h_t_cnt = max_h_t_cnt + 1
                else:
                    ht_index = triple_dict[(h,t)]
                    relation_multi_label[k, ht_index, r] = 1
                    continue

                vertexSet = copy.deepcopy(ins['vertexSet'])
                evidence_sents = copy.deepcopy(triple['evidence'])

                words = []

                for sent in evidence_sents:
                    words += ins['sents'][sent]

                max_local_wn = max(len(words), max_local_wn)

                for t,word in enumerate(words):
                    word = word.lower()
                    if t < MAX_LENGTH:
                        if word in word2id:
                            local_context_idxs[k, j, t] = word2id[word]
                        else:
                            local_context_idxs[k, j, t] = word2id['UNK']

                for t in range(t+1, MAX_LENGTH):
                    local_context_idxs[k, j, t] = word2id['BLANK']

                Ls = copy.deepcopy(ins['Ls'])

                pos_idx = 1
                pos_flag = False
                for idx, vertex in enumerate(vertexSet):
                    for v in vertex:
                        sent_id = int(v['sent_id'])
                        if sent_id not in evidence_sents:
                            continue

                        pos_flag = True
                        dl = 0
                        for s_i in range(sent_id):
                            if s_i not in evidence_sents:
                                dl = dl + Ls[s_i]

                        local_context_pos[k, j, v['pos'][0] - dl:v['pos'][1] - dl] = pos_idx
                        local_context_ner[k, j, v['pos'][0] - dl:v['pos'][1] - dl] = ner2id[v['type']]

                    if pos_flag:
                        pos_idx = pos_idx + 1
                        pos_flag = False

                h_list = vertexSet[triple['h']]
                t_list = vertexSet[triple['t']]

                for h_i, h in enumerate(h_list):
                    global_h_mapping[k, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(h_list) / (h['pos'][1] - h['pos'][0])

                for t_i, t in enumerate(t_list):
                    global_t_mapping[k, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(t_list) / (t['pos'][1] - t['pos'][0])

                delta_dis = h_list[0]['pos'][0] - t_list[0]['pos'][0]
                if delta_dis < 0:
                    global_ht_pos[k, j] = -int(dis2idx[-delta_dis])
                else:
                    global_ht_pos[k, j] = int(dis2idx[delta_dis])

                h_num = 0
                h_dl = []
                for h in h_list:
                    sent_id = int(h['sent_id'])
                    if sent_id in evidence_sents:
                        h_num = h_num + 1

                    dl = 0
                    for s_i in range(sent_id):
                        if s_i not in evidence_sents:
                            dl = dl + Ls[s_i]
                    h_dl.append(dl)

                t_num = 0
                t_dl = []
                for t in t_list:
                    sent_id = t['sent_id']
                    if sent_id in evidence_sents:
                        t_num = t_num + 1

                    dl = 0
                    for s_i in range(sent_id):
                        if s_i not in evidence_sents:
                            dl = dl + Ls[s_i]
                    t_dl.append(dl)

                for h_i, h in enumerate(h_list):
                    if h['sent_id'] not in evidence_sents:
                        continue
                    local_h_mapping[k, j, h['pos'][0]-h_dl[h_i]:h['pos'][1]-h_dl[h_i]] = 1.0 / h_num / (h['pos'][1] - h['pos'][0])

                for t_i, t in enumerate(t_list):
                    if t['sent_id'] not in evidence_sents:
                        continue
                    local_t_mapping[k, j, t['pos'][0]-t_dl[t_i]:t['pos'][1]-t_dl[t_i]] = 1.0 / t_num / (t['pos'][1] - t['pos'][0])

                h_in_evidence, t_in_evidence = False, False
                for h_idex, h in enumerate(h_list):
                    sent_id = int(h['sent_id'])
                    if sent_id in evidence_sents:
                        dis_h = h
                        dis_h_idx = h_idex
                        h_in_evidence = True
                        break
                for t_idx, t in enumerate(t_list):
                    sent_id = int(t['sent_id'])
                    if sent_id in evidence_sents:
                        dis_t = t
                        dis_t_idx = t_idx
                        t_in_evidence = True
                        break

                if h_in_evidence & t_in_evidence:
                    delta_dis = (dis_h['pos'][0]-h_dl[dis_h_idx]) - (dis_t['pos'][0]-t_dl[dis_t_idx])
                else:
                    delta_dis = 0

                if delta_dis < 0:
                    local_ht_pos[k, j] = -int(dis2idx[-delta_dis])
                else:
                    local_ht_pos[k,j] = int(dis2idx[delta_dis])

                relation_mask[k, j] = 1
                relation_label[k, j] = r

            max_h_t_cnt = max(max_h_t_cnt, h_t_cnt)

        yield {'global_context_idxs': global_context_idxs[:cur_index, :max_global_wn],
               'global_context_pos': global_context_pos[:cur_index, :max_global_wn],
               'global_context_ner': global_context_ner[:cur_index, :max_global_wn],
               'global_h_mapping': global_h_mapping[:cur_index, :max_h_t_cnt, :max_global_wn],
               'global_t_mapping': global_t_mapping[:cur_index, :max_h_t_cnt, :max_global_wn],
               'global_ht_pos': global_ht_pos[:cur_index, :max_h_t_cnt],
               'local_context_idxs': local_context_idxs[:cur_index, :max_h_t_cnt, :max_local_wn],
               'local_context_pos': local_context_pos[:cur_index, :max_h_t_cnt, :max_local_wn],
               'local_context_ner': local_context_ner[:cur_index, :max_h_t_cnt, :max_local_wn],
               'local_h_mapping': local_h_mapping[:cur_index, :max_h_t_cnt, :max_local_wn],
               'local_t_mapping': local_t_mapping[:cur_index, :max_h_t_cnt, :max_local_wn],
               'local_ht_pos': local_ht_pos[:cur_index, :max_h_t_cnt],
               'relation_label': relation_label[:cur_index, :max_h_t_cnt],
               'relation_multi_label': relation_multi_label[:cur_index, :max_h_t_cnt, :],
               'relation_mask': relation_mask[:cur_index, :max_h_t_cnt],
               'max_global_wn': max_global_wn,
               'max_local_wn': max_local_wn,
               'max_h_t_cnt': max_h_t_cnt,
        }

def process_dev_data(dev_file, word2id, ner2id):
    global dis2idx
    train_prefix = 'dev_train'
    test_data_size = len(dev_file)

    batch_num = test_data_size // TEST_BATCH_SIZE
    if test_data_size % TEST_BATCH_SIZE != 0:
        batch_num = batch_num + 1

    test_order = list(range(test_data_size))

    for i in range(batch_num):
        start_index = i * TEST_BATCH_SIZE
        cur_index = min(TEST_BATCH_SIZE, test_data_size - start_index)
        cur_order = list(test_order[start_index : start_index+cur_index])

        global_context_idxs = np.zeros((cur_index, MAX_LENGTH), dtype=np.int32)
        global_context_pos = np.zeros((cur_index, MAX_LENGTH), dtype=np.int32)
        global_context_ner = np.zeros((cur_index, MAX_LENGTH), dtype=np.int32)
        global_h_mapping = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.float32)
        global_t_mapping = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.float32)
        global_ht_pos = np.zeros((cur_index, H_T_LIMIT), dtype=np.int32)

        local_context_idxs = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.int32)
        local_context_pos = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.int32)
        local_context_ner = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.int32)
        local_h_mapping = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.float32)
        local_t_mapping = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.float32)
        local_ht_pos = np.zeros((cur_index, H_T_LIMIT), dtype=np.int32)

        relation_multi_label = np.zeros((cur_index, H_T_LIMIT, CLASS_NUM), dtype=np.float32)
        relation_mask = np.zeros((cur_index, H_T_LIMIT), dtype=np.float32)

        titles = []
        L_labels = []
        L_vertex = []
        indexes = []
        all_ht_triple = []

        max_h_t_cnt = 0
        max_global_wn = 0
        max_local_wn = 0

        for k, index in enumerate(cur_order):
            ins = dev_file[index]

            words = []
            for sent in ins['sents']:
                words += sent

            max_global_wn = max(len(words), max_global_wn)
            for t, word in enumerate(words):
                word = word.lower()
                if t < MAX_LENGTH:
                    if word in word2id:
                        global_context_idxs[k, t] = word2id[word]
                    else:
                        global_context_idxs[k, t] = word2id['UNK']

            for t in range(t + 1, MAX_LENGTH):
                global_context_idxs[k, t] = word2id['BLANK']

            vertexSet = copy.deepcopy(ins['vertexSet'])
            for idx, vertex in enumerate(vertexSet, 1):
                for v in vertex:
                    global_context_pos[k, v['pos'][0]:v['pos'][1]] = idx
                    global_context_ner[k, v['pos'][0]:v['pos'][1]] = ner2id[v['type']]

            all_triples = copy.deepcopy(ins['all_triples'])
            random.shuffle(all_triples)

            h_t_cnt = 0

            ht_triple = []

            label_set = {}

            triple_dict = dict()

            for j, triple in enumerate(all_triples):
                h_ = triple['h']
                t_ = triple['t']
                r_ = triple['r']

                if (h_, t_) not in triple_dict.keys():
                    triple_dict[(h_, t_)] = j
                    relation_multi_label[k, j, r_] = 1
                    max_h_t_cnt = max_h_t_cnt + 1
                else:
                    ht_index = triple_dict[(h_, t_)]
                    relation_multi_label[k, ht_index, r_] = 1
                    continue

                vertexSet = copy.deepcopy(ins['vertexSet'])

                evidence_sents = copy.deepcopy(triple['evidence'])
                words = []

                for sent in evidence_sents:
                    words += ins['sents'][sent]

                max_local_wn = max(len(words), max_local_wn)

                for t, word in enumerate(words):
                    word = word.lower()
                    if t < MAX_LENGTH:
                        if word in word2id:
                            local_context_idxs[k, j, t] = word2id[word]
                        else:
                            local_context_idxs[k, j, t] = word2id['UNK']

                for t in range(t + 1, MAX_LENGTH):
                    local_context_idxs[k, j, t] = word2id['BLANK']

                Ls = copy.deepcopy(ins['Ls'])

                pos_idx = 1
                pos_flag = False
                for idx, vertex in enumerate(vertexSet):
                    for v in vertex:
                        sent_id = int(v['sent_id'])
                        if sent_id not in evidence_sents:
                            continue

                        pos_flag = True
                        dl = 0
                        for s_i in range(sent_id):
                            if s_i not in evidence_sents:
                                dl = dl + Ls[s_i]

                        local_context_pos[k, j, v['pos'][0] - dl:v['pos'][1] - dl] = pos_idx
                        local_context_ner[k, j, v['pos'][0] - dl:v['pos'][1] - dl] = ner2id[v['type']]

                    if pos_flag:
                        pos_idx = pos_idx + 1
                        pos_flag = False

                h_list = vertexSet[triple['h']]
                t_list = vertexSet[triple['t']]

                for h_i, h in enumerate(h_list):
                    global_h_mapping[k, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(h_list) / (h['pos'][1] - h['pos'][0])

                for t_i, t in enumerate(t_list):
                    global_t_mapping[k, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(t_list) / (t['pos'][1] - t['pos'][0])

                delta_dis = h_list[0]['pos'][0] - t_list[0]['pos'][0]
                if delta_dis < 0:
                    global_ht_pos[k, j] = -int(dis2idx[-delta_dis])
                else:
                    global_ht_pos[k, j] = int(dis2idx[delta_dis])

                h_num = 0
                h_dl = []
                for h in h_list:
                    sent_id = int(h['sent_id'])
                    if sent_id in evidence_sents:
                        h_num = h_num + 1

                    dl = 0
                    for s_i in range(sent_id):
                        if s_i not in evidence_sents:
                            dl = dl + Ls[s_i]
                    h_dl.append(dl)

                t_num = 0
                t_dl = []
                for t in t_list:
                    sent_id = t['sent_id']
                    if sent_id in evidence_sents:
                        t_num = t_num + 1

                    dl = 0
                    for s_i in range(sent_id):
                        if s_i not in evidence_sents:
                            dl = dl + Ls[s_i]
                    t_dl.append(dl)

                for h_i, h in enumerate(h_list):
                    if h['sent_id'] not in evidence_sents:
                        continue
                    local_h_mapping[k, j, h['pos'][0] - h_dl[h_i]:h['pos'][1] - h_dl[h_i]] = 1.0 / h_num / (
                                h['pos'][1] - h['pos'][0])

                for t_i, t in enumerate(t_list):
                    if t['sent_id'] not in evidence_sents:
                        continue
                    local_t_mapping[k, j, t['pos'][0] - t_dl[t_i]:t['pos'][1] - t_dl[t_i]] = 1.0 / t_num / (
                                t['pos'][1] - t['pos'][0])

                h_in_evidence, t_in_evidence = False, False
                for h_idex, h in enumerate(h_list):
                    sent_id = int(h['sent_id'])
                    if sent_id in evidence_sents:
                        dis_h = h
                        dis_h_idx = h_idex
                        h_in_evidence = True
                        break
                for t_idx, t in enumerate(t_list):
                    sent_id = int(t['sent_id'])
                    if sent_id in evidence_sents:
                        dis_t = t
                        dis_t_idx = t_idx
                        t_in_evidence = True
                        break

                if h_in_evidence & t_in_evidence:
                    delta_dis = (dis_h['pos'][0]-h_dl[dis_h_idx]) - (dis_t['pos'][0]-t_dl[dis_t_idx])
                else:
                    delta_dis = 0

                if delta_dis < 0:
                    local_ht_pos[k, j] = -int(dis2idx[-delta_dis])
                else:
                    local_ht_pos[k,j] = int(dis2idx[delta_dis])

                r = triple['r']
                relation_mask[k, j] = 1
                if r != 0:
                    label_set[(triple['h'], triple['t'], triple['r'])] = triple['in' + train_prefix]

                ht_triple.append((triple['h'], triple['t']))

                h_t_cnt = h_t_cnt + 1


            max_h_t_cnt = max(max_h_t_cnt, h_t_cnt)

            L = len(ins['vertexSet'])
            L_vertex.append(L)
            L_labels.append(label_set)
            titles.append(ins['title'])
            indexes.append(index)
            all_ht_triple.append(ht_triple)

        yield {'global_context_idxs': global_context_idxs[:cur_index, :max_global_wn],
               'global_context_pos': global_context_pos[:cur_index, :max_global_wn],
               'global_context_ner': global_context_ner[:cur_index, :max_global_wn],
               'global_h_mapping': global_h_mapping[:cur_index, :max_h_t_cnt, :max_global_wn],
               'global_t_mapping': global_t_mapping[:cur_index, :max_h_t_cnt, :max_global_wn],
               'global_ht_pos': global_ht_pos[:cur_index, :max_h_t_cnt],
               'local_context_idxs': local_context_idxs[:cur_index, :max_h_t_cnt, :max_local_wn],
               'local_context_pos': local_context_pos[:cur_index, :max_h_t_cnt, :max_local_wn],
               'local_context_ner': local_context_ner[:cur_index, :max_h_t_cnt, :max_local_wn],
               'local_h_mapping': local_h_mapping[:cur_index, :max_h_t_cnt, :max_local_wn],
               'local_t_mapping': local_t_mapping[:cur_index, :max_h_t_cnt, :max_local_wn],
               'local_ht_pos': local_ht_pos[:cur_index, :max_h_t_cnt],
               'labels': L_labels,
               'L_vertex': L_vertex,
               'relation_mask': relation_mask[:cur_index, :max_h_t_cnt],
               'relation_multi_label': relation_multi_label[:cur_index, :max_h_t_cnt, :],
               'titles': titles,
               'indexes': indexes,
               'max_global_wn': max_global_wn,
               'max_local_wn': max_local_wn,
               'max_h_t_cnt': max_h_t_cnt,
               'all_ht_triple': all_ht_triple,
               }

def process_test_data(test_file, word2id, ner2id):
    global dis2idx
    train_prefix = 'dev_train'
    test_data_size = len(test_file)

    batch_num = test_data_size // TEST_BATCH_SIZE
    if test_data_size % TEST_BATCH_SIZE != 0:
        batch_num = batch_num + 1

    test_order = list(range(test_data_size))

    for i in range(batch_num):
        start_index = i * TEST_BATCH_SIZE
        cur_index = min(TEST_BATCH_SIZE, test_data_size - start_index)
        cur_order = list(test_order[start_index : start_index+cur_index])

        global_context_idxs = np.zeros((cur_index, MAX_LENGTH), dtype=np.int32)
        global_context_pos = np.zeros((cur_index, MAX_LENGTH), dtype=np.int32)
        global_context_ner = np.zeros((cur_index, MAX_LENGTH), dtype=np.int32)
        global_h_mapping = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.float32)
        global_t_mapping = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.float32)
        global_ht_pos = np.zeros((cur_index, H_T_LIMIT), dtype=np.int32)

        local_context_idxs = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.int32)
        local_context_pos = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.int32)
        local_context_ner = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.int32)
        local_h_mapping = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.float32)
        local_t_mapping = np.zeros((cur_index, H_T_LIMIT, MAX_LENGTH), dtype=np.float32)
        local_ht_pos = np.zeros((cur_index, H_T_LIMIT), dtype=np.int32)

        relation_mask = np.zeros((cur_index, H_T_LIMIT), dtype=np.float32)

        titles = []
        L_labels = []
        L_vertex = []
        indexes = []
        all_ht_triple = []

        max_h_t_cnt = 0
        max_global_wn = 0
        max_local_wn = 0

        for k, index in enumerate(cur_order):
            ins = test_file[index]

            words = []
            for sent in ins['sents']:
                words += sent

            max_global_wn = max(len(words), max_global_wn)
            for t, word in enumerate(words):
                word = word.lower()
                if t < MAX_LENGTH:
                    if word in word2id:
                        global_context_idxs[k, t] = word2id[word]
                    else:
                        global_context_idxs[k, t] = word2id['UNK']

            for t in range(t + 1, MAX_LENGTH):
                global_context_idxs[k, t] = word2id['BLANK']

            vertexSet = copy.deepcopy(ins['vertexSet'])
            for idx, vertex in enumerate(vertexSet, 1):
                for v in vertex:
                    global_context_pos[k, v['pos'][0]:v['pos'][1]] = idx
                    global_context_ner[k, v['pos'][0]:v['pos'][1]] = ner2id[v['type']]

            all_triples = copy.deepcopy(ins['all_triples'])
            random.shuffle(all_triples)

            h_t_cnt = 0

            ht_triple = []

            label_set = {}

            for j, triple in enumerate(all_triples):
                vertexSet = copy.deepcopy(ins['vertexSet'])

                evidence_sents = copy.deepcopy(triple['evidence'])
                words = []

                for sent in evidence_sents:
                    words += ins['sents'][sent]

                max_local_wn = max(len(words), max_local_wn)

                for t, word in enumerate(words):
                    word = word.lower()
                    if t < MAX_LENGTH:
                        if word in word2id:
                            local_context_idxs[k, j, t] = word2id[word]
                        else:
                            local_context_idxs[k, j, t] = word2id['UNK']

                for t in range(t + 1, MAX_LENGTH):
                    local_context_idxs[k, j, t] = word2id['BLANK']

                Ls = copy.deepcopy(ins['Ls'])

                pos_idx = 1
                pos_flag = False
                for idx, vertex in enumerate(vertexSet):
                    for v in vertex:
                        sent_id = int(v['sent_id'])
                        if sent_id not in evidence_sents:
                            continue

                        pos_flag = True
                        dl = 0
                        for s_i in range(sent_id):
                            if s_i not in evidence_sents:
                                dl = dl + Ls[s_i]

                        local_context_pos[k, j, v['pos'][0] - dl:v['pos'][1] - dl] = pos_idx
                        local_context_ner[k, j, v['pos'][0] - dl:v['pos'][1] - dl] = ner2id[v['type']]

                    if pos_flag:
                        pos_idx = pos_idx + 1
                        pos_flag = False

                h_list = vertexSet[triple['h']]
                t_list = vertexSet[triple['t']]

                for h_i, h in enumerate(h_list):
                    global_h_mapping[k, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(h_list) / (h['pos'][1] - h['pos'][0])

                for t_i, t in enumerate(t_list):
                    global_t_mapping[k, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(t_list) / (t['pos'][1] - t['pos'][0])

                delta_dis = h_list[0]['pos'][0] - t_list[0]['pos'][0]
                if delta_dis < 0:
                    global_ht_pos[k, j] = -int(dis2idx[-delta_dis])
                else:
                    global_ht_pos[k, j] = int(dis2idx[delta_dis])

                h_num = 0
                h_dl = []
                for h in h_list:
                    sent_id = int(h['sent_id'])
                    if sent_id in evidence_sents:
                        h_num = h_num + 1

                    dl = 0
                    for s_i in range(sent_id):
                        if s_i not in evidence_sents:
                            dl = dl + Ls[s_i]
                    h_dl.append(dl)

                t_num = 0
                t_dl = []
                for t in t_list:
                    sent_id = t['sent_id']
                    if sent_id in evidence_sents:
                        t_num = t_num + 1

                    dl = 0
                    for s_i in range(sent_id):
                        if s_i not in evidence_sents:
                            dl = dl + Ls[s_i]
                    t_dl.append(dl)

                for h_i, h in enumerate(h_list):
                    if h['sent_id'] not in evidence_sents:
                        continue
                    local_h_mapping[k, j, h['pos'][0] - h_dl[h_i]:h['pos'][1] - h_dl[h_i]] = 1.0 / h_num / (
                                h['pos'][1] - h['pos'][0])

                for t_i, t in enumerate(t_list):
                    if t['sent_id'] not in evidence_sents:
                        continue
                    local_t_mapping[k, j, t['pos'][0] - t_dl[t_i]:t['pos'][1] - t_dl[t_i]] = 1.0 / t_num / (
                                t['pos'][1] - t['pos'][0])

                h_in_evidence, t_in_evidence = False, False
                for h_idex, h in enumerate(h_list):
                    sent_id = int(h['sent_id'])
                    if sent_id in evidence_sents:
                        dis_h = h
                        dis_h_idx = h_idex
                        h_in_evidence = True
                        break
                for t_idx, t in enumerate(t_list):
                    sent_id = int(t['sent_id'])
                    if sent_id in evidence_sents:
                        dis_t = t
                        dis_t_idx = t_idx
                        t_in_evidence = True
                        break

                if h_in_evidence & t_in_evidence:
                    delta_dis = (dis_h['pos'][0]-h_dl[dis_h_idx]) - (dis_t['pos'][0]-t_dl[dis_t_idx])
                else:
                    delta_dis = 0

                if delta_dis < 0:
                    local_ht_pos[k, j] = -int(dis2idx[-delta_dis])
                else:
                    local_ht_pos[k,j] = int(dis2idx[delta_dis])

                r = triple['r']
                relation_mask[k, j] = 1
                if r != 0:
                    label_set[(triple['h'], triple['t'], triple['r'])] = triple['in' + train_prefix]

                ht_triple.append((triple['h'], triple['t']))

                h_t_cnt = h_t_cnt + 1


            max_h_t_cnt = max(max_h_t_cnt, h_t_cnt)

            L = len(ins['vertexSet'])
            L_vertex.append(L)
            L_labels.append(label_set)
            titles.append(ins['title'])
            indexes.append(index)
            all_ht_triple.append(ht_triple)

        yield {'global_context_idxs': global_context_idxs[:cur_index, :max_global_wn],
               'global_context_pos': global_context_pos[:cur_index, :max_global_wn],
               'global_context_ner': global_context_ner[:cur_index, :max_global_wn],
               'global_h_mapping': global_h_mapping[:cur_index, :max_h_t_cnt, :max_global_wn],
               'global_t_mapping': global_t_mapping[:cur_index, :max_h_t_cnt, :max_global_wn],
               'global_ht_pos': global_ht_pos[:cur_index, :max_h_t_cnt],
               'local_context_idxs': local_context_idxs[:cur_index, :max_h_t_cnt, :max_local_wn],
               'local_context_pos': local_context_pos[:cur_index, :max_h_t_cnt, :max_local_wn],
               'local_context_ner': local_context_ner[:cur_index, :max_h_t_cnt, :max_local_wn],
               'local_h_mapping': local_h_mapping[:cur_index, :max_h_t_cnt, :max_local_wn],
               'local_t_mapping': local_t_mapping[:cur_index, :max_h_t_cnt, :max_local_wn],
               'local_ht_pos': local_ht_pos[:cur_index, :max_h_t_cnt],
               'labels': L_labels,
               'L_vertex': L_vertex,
               'relation_mask': relation_mask[:cur_index, :max_h_t_cnt],
               'titles': titles,
               'indexes': indexes,
               'max_global_wn': max_global_wn,
               'max_local_wn': max_local_wn,
               'max_h_t_cnt': max_h_t_cnt,
               'all_ht_triple': all_ht_triple,
               }

def train(train_file, test_file, CModel, id2rel, word2id, ner2id, MODEL_PATH, TRAIN_EPOCHES, TEST_EPOCH, CL_LEARNING_RATE, postfix):
    acc_na = Accuracy()
    acc_not_na = Accuracy()
    acc_total = Accuracy()

    train_loss = []
    dev_loss = []

    best_f1 = float('-inf')

    for epoch in range(TRAIN_EPOCHES):

        acc_na.clear()
        acc_not_na.clear()
        acc_total.clear()
        total_loss = 0.0
        step = 0

        for data in process_train_data(train_file, word2id, ner2id):

            relation_label = data['relation_label']

            pre, loss = CModel.train(data, CL_LEARNING_RATE)

            total_loss = total_loss + loss

            for i in range(pre.shape[0]):
                for k in range(pre.shape[1]):
                    label = relation_label[i][k]
                    if label < 0:
                        break

                    if label == 0:
                        acc_na.add(pre[i][k] == label)
                    else:
                        acc_not_na.add(pre[i][k] == label)

                    acc_total.add(pre[i][k] == label)

            step = step + 1

        epoch_loss = total_loss
        train_loss.append(epoch_loss)
        print("train epoch: %s; train loss: %s; accuracy na: %s; accuracy not na: %s; accuracy: total: %s" \
              % (str(epoch), str(epoch_loss), str(acc_na.get()), str(acc_not_na.get()), str(acc_total.get())))

        total_loss = 0.0

        if epoch % TEST_EPOCH == 0:
            test_result = []
            total_recall = 0
            total_recall_ignore = 0
            top1_acc = have_label = 0
            input_theta = -1

            dev_total_loss = 0.0
            step = 0

            for test_data in process_dev_data(test_file, word2id, ner2id):

                labels = test_data['labels']
                L_vertex = test_data['L_vertex']
                indexes = test_data['indexes']
                titles = test_data['titles']
                all_ht_triple = test_data['all_ht_triple']

                pre, loss = CModel.test(test_data)
                dev_total_loss = dev_total_loss + loss

                for i in range(len(labels)):
                    label = labels[i]
                    index = indexes[i]
                    ht_triple = all_ht_triple[i]

                    total_recall = total_recall + len(label)
                    for l in label.values():
                        if not l:
                            total_recall_ignore += 1

                    L = L_vertex[i]  # entity mention num
                    k = 0

                    for triple in ht_triple:
                        h_idx = triple[0]
                        t_idx = triple[1]
                        r = np.argmax(pre[i, k])
                        if (h_idx, t_idx, r) in label:
                            top1_acc = top1_acc + 1

                        flag = False

                        for r in range(1, CLASS_NUM):
                            in_train = False

                            if (h_idx, t_idx, r) in label:
                                flag = True
                                if label[(h_idx, t_idx, r)] == True:
                                    in_train = True

                            test_result.append(((h_idx, t_idx, r) in label, float(pre[i, k, r]), in_train, titles[i],
                                                id2rel[r], index, h_idx, t_idx, r))

                        if flag:
                            have_label = have_label + 1

                        k = k + 1

                step = step + 1

            test_result.sort(key=lambda x: x[1], reverse=True)
            print('total_recall: %s' % str(total_recall))

            pr_x = []
            pr_y = []
            correct = 0
            w = 0

            if total_recall == 0:
                total_recall = 1

            for i, item in enumerate(test_result):
                correct = correct + item[0]  # true exist h and t relation num
                pr_y.append(float(correct) / (i + 1))
                pr_x.append(float(correct) / total_recall)
                if item[1] > input_theta:
                    w = i

            pr_x = np.asarray(pr_x, dtype='float32')
            pr_y = np.asarray(pr_y, dtype='float32')
            f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
            f1 = f1_arr.max()
            f1_pos = f1_arr.argmax()
            theta = test_result[f1_pos][1]

            if input_theta == -1:
                w = f1_pos
                input_theta = theta

            auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
            print("ALL : Theta {:%3.4f} | F1 {:%3.4f} | AUC {:%3.4f}" % (theta, f1, auc))

            all_f1 = f1

            pr_x = []
            pr_y = []
            correct = 0
            correct_in_train = 0
            w = 0
            for i, item in enumerate(test_result):
                correct = correct + item[0]
                if item[0] & item[2]:
                    correct_in_train = correct_in_train + 1
                if correct_in_train == correct:
                    p = 0
                else:
                    p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
                pr_y.append(p)
                pr_x.append(float(correct) / total_recall)
                if item[1] > input_theta:
                    w = i

            pr_x = np.asarray(pr_x, dtype='float32')
            pr_y = np.asarray(pr_y, dtype='float32')
            f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
            f1 = f1_arr.max()

            auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
            print("Ignore ma_f1 {:%3.4f} | input_theta {:%3.4f} test_result F1 {:%3.4f} | AUC {:%3.4f}" % (
            f1, input_theta, f1_arr[w], auc))

            if all_f1 > best_f1:
                save_model_path = MODEL_PATH + '/' + 'ClassModel' + '/' + 'best' + '/' + postfix + '_' + 'cmodel.ckpt'
                CModel.save(save_model_path)
                best_f1 = max(best_f1, all_f1)


            # dev_epoch_loss = dev_total_loss / step
            dev_epoch_loss = dev_total_loss
            dev_loss.append(dev_epoch_loss)

    # model_path = MODEL_PATH + '/' + 'ClassModel/cmodel.ckpt'
    # CModel.save(model_path)

    if PRINT_LOSS:
        train_loss_path = 'output' + '/' + postfix + '_' + 'train_loss.csv'
        with open(train_loss_path, 'a', encoding='utf-8') as f_write:
            for i, loss in enumerate(train_loss):
                f_write.write(str(i))
                f_write.write('\t')
                f_write.write(str(loss))
                f_write.write('\n')
        f_write.close()

        dev_loss_path = 'output' + '/' + postfix + '_' + 'dev_loss.csv'
        with open(dev_loss_path, 'a', encoding='utf-8') as f_write:
            for i, loss in enumerate(dev_loss):
                f_write.write(str(i))
                f_write.write('\t')
                f_write.write(str(loss))
                f_write.write('\n')
        f_write.close()


    return

def main(DATH_PATH, CModel, id2rel, word2id, ner2id, MODEL_PATH, TRAIN_EPOCHES, TEST_EPOCH, CL_LEARNING_RATE, POST_FIX='evidence'):
    train_file, test_file = read_train_data(DATA_PATH=DATH_PATH)

    train(train_file, test_file, CModel, id2rel, word2id, ner2id, MODEL_PATH, TRAIN_EPOCHES, TEST_EPOCH, CL_LEARNING_RATE, POST_FIX)

    return