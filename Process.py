import re
import copy
import numpy as np

dis2idx = np.zeros((512), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9

class Process():
    def __init__(self, MAX_LENGTH, HIS_MAX_LENGTH, CUR_MAX_LENGTH, H_T_LIMIT, CLASS_NUM, word2id, ner2id):
        self.MAX_LENGTH = MAX_LENGTH
        self.HIS_MAX_LENGTH = HIS_MAX_LENGTH
        self.CUR_MAX_LENGTH = CUR_MAX_LENGTH
        self.H_T_LIMIT = H_T_LIMIT
        self.CLASS_NUM = CLASS_NUM
        self.word2id = word2id
        self.ner2id = ner2id

        return

    def process_his_data(self, data, label, evidence_sents):
        sents = copy.deepcopy(data['sents'])
        vertexSet = copy.deepcopy(data['vertexSet'])
        Ls = copy.deepcopy(data['Ls'])

        context_idxs = np.zeros((1, self.HIS_MAX_LENGTH), dtype=np.int64)
        context_pos = np.zeros((1, self.HIS_MAX_LENGTH), dtype=np.int64)
        context_ner = np.zeros((1, self.HIS_MAX_LENGTH), dtype=np.int64)

        h_mapping = np.zeros((1, 1, self.HIS_MAX_LENGTH), dtype=np.float32)
        t_mapping = np.zeros((1, 1, self.HIS_MAX_LENGTH), dtype=np.float32)

        words = []
        for sent_id in evidence_sents:
            words += sents[sent_id]

        for t, word in enumerate(words):
            word = word.lower()
            if t < self.HIS_MAX_LENGTH:
                if word in self.word2id:
                    context_idxs[0, t] = self.word2id[word]
                else:
                    context_idxs[0, t] = self.word2id['UNK']

        for t in range(t+1, self.HIS_MAX_LENGTH):
            context_idxs[0, t] = self.word2id['BLANK']

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

                context_pos[0, v['pos'][0] - dl:v['pos'][1] - dl] = pos_idx
                context_ner[0, v['pos'][0] - dl:v['pos'][1] - dl] = self.ner2id[v['type']]

            if pos_flag:
                pos_idx = pos_idx + 1
                pos_flag = False

        h_list = vertexSet[label['h']]
        t_list = vertexSet[label['t']]

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
            h_mapping[0, 0, h['pos'][0] - h_dl[h_i]:h['pos'][1] - h_dl[h_i]] = 1.0 / h_num / (h['pos'][1] - h['pos'][0])

        for t_i, t in enumerate(t_list):
            if t['sent_id'] not in evidence_sents:
                continue
            t_mapping[0, 0, t['pos'][0] - t_dl[t_i]:t['pos'][1] - t_dl[t_i]] = 1.0 / t_num / (t['pos'][1] - t['pos'][0])

        return context_idxs, context_pos, context_ner, h_mapping, t_mapping

    def process_cur_data(self, data, sent_id):
        sents = copy.deepcopy(data['sents'])
        vertexSet = copy.deepcopy(data['vertexSet'])

        sen_idx = np.zeros((1, self.CUR_MAX_LENGTH), dtype=np.int64)
        sen_pos = np.zeros((1, self.CUR_MAX_LENGTH), dtype=np.int64)
        sen_ner = np.zeros((1, self.CUR_MAX_LENGTH), dtype=np.int64)

        words = sents[sent_id]

        for t, word in enumerate(words):
            word = word.lower()
            if t < self.CUR_MAX_LENGTH:
                if word in self.word2id:
                    sen_idx[0, t] = self.word2id[word]
                else:
                    sen_idx[0, t] = self.word2id['UNK']

        for t in range(t + 1, self.CUR_MAX_LENGTH):
            sen_idx[0, t] = self.word2id['BLANK']

        Ls = copy.deepcopy(data['Ls'])

        pos_idx = 1
        pos_flag = False
        for ids, vertex in enumerate(vertexSet):
            for v in vertex:
                v_send_id = int(v['sent_id'])
                if v_send_id != sent_id:
                    continue

                pos_flag = True
                dl = 0
                for s_i in range(v_send_id):
                    if s_i != sent_id:
                        dl = dl + Ls[s_i]

                sen_pos[0, v['pos'][0] - dl:v['pos'][1] - dl] = pos_idx
                sen_ner[0, v['pos'][0] - dl:v['pos'][1] - dl] = self.ner2id[v['type']]

            if pos_flag:
                pos_idx = pos_idx + 1
                pos_flag = False

        return sen_idx, sen_pos, sen_ner

    def process_cldata(self, ins):
        global_context_idxs = np.zeros((1, self.MAX_LENGTH), dtype=np.int32)
        global_context_pos = np.zeros((1, self.MAX_LENGTH), dtype=np.int32)
        global_context_ner = np.zeros((1, self.MAX_LENGTH), dtype=np.int32)
        global_h_mapping = np.zeros((1, self.H_T_LIMIT, self.MAX_LENGTH), dtype=np.float32)
        global_t_mapping = np.zeros((1, self.H_T_LIMIT, self.MAX_LENGTH), dtype=np.float32)
        global_ht_pos = np.zeros((1, self.H_T_LIMIT), dtype=np.int32)

        local_context_idxs = np.zeros((1, self.H_T_LIMIT, self.MAX_LENGTH), dtype=np.int32)
        local_context_pos = np.zeros((1, self.H_T_LIMIT, self.MAX_LENGTH), dtype=np.int32)
        local_context_ner = np.zeros((1, self.H_T_LIMIT, self.MAX_LENGTH), dtype=np.int32)
        local_h_mapping = np.zeros((1, self.H_T_LIMIT, self.MAX_LENGTH), dtype=np.float32)
        local_t_mapping = np.zeros((1, self.H_T_LIMIT, self.MAX_LENGTH), dtype=np.float32)
        local_ht_pos = np.zeros((1, self.H_T_LIMIT), dtype=np.int32)

        relation_multi_label = np.zeros((1, self.H_T_LIMIT, self.CLASS_NUM), dtype=np.float32)
        relation_label = np.full([1, self.H_T_LIMIT], fill_value=-100)
        relation_mask = np.zeros((1, self.H_T_LIMIT), dtype=np.float32)

        max_global_wn = 0
        max_local_wn = 0
        max_h_t_cnt = 0

        words = []
        for sent in ins['sents']:
            words += sent

        max_global_wn = max(len(words), max_global_wn)
        for t, word in enumerate(words):
            word = word.lower()
            if t < self.MAX_LENGTH:
                if word in self.word2id:
                    global_context_idxs[0, t] = self.word2id[word]
                else:
                    global_context_idxs[0, t] = self.word2id['UNK']

        for t in range(t + 1, self.MAX_LENGTH):
            global_context_idxs[0, t] = self.word2id['BLANK']

        vertexSet = copy.deepcopy(ins['vertexSet'])
        for idx, vertex in enumerate(vertexSet, 1):
            for v in vertex:
                global_context_pos[0, v['pos'][0]:v['pos'][1]] = idx
                global_context_ner[0, v['pos'][0]:v['pos'][1]] = self.ner2id[v['type']]

        labels = copy.deepcopy(ins['labels'])

        for j, label in enumerate(labels, 0):
            vertexSet = copy.deepcopy(ins['vertexSet'])

            evidence_sents = copy.deepcopy(label['evidence'])
            words = []

            for sent in evidence_sents:
                words += ins['sents'][sent]

            max_local_wn = max(len(words), max_local_wn)

            for t, word in enumerate(words):
                word = word.lower()
                if t < self.MAX_LENGTH:
                    if word in self.word2id:
                        local_context_idxs[0, j, t] = self.word2id[word]
                    else:
                        local_context_idxs[0, j, t] = self.word2id['UNK']

            for t in range(t + 1, self.MAX_LENGTH):
                local_context_idxs[0, j, t] = self.word2id['BLANK']

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

                    local_context_pos[0, j, v['pos'][0] - dl:v['pos'][1] - dl] = pos_idx
                    local_context_ner[0, j, v['pos'][0] - dl:v['pos'][1] - dl] = self.ner2id[v['type']]

                if pos_flag:
                    pos_idx = pos_idx + 1
                    pos_flag = False

            h_list = vertexSet[label['h']]
            t_list = vertexSet[label['t']]

            for h_i, h in enumerate(h_list):
                global_h_mapping[0, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(h_list) / (h['pos'][1] - h['pos'][0])

            for t_i, t in enumerate(t_list):
                global_t_mapping[0, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(t_list) / (t['pos'][1] - t['pos'][0])

            delta_dis = h_list[0]['pos'][0] - t_list[0]['pos'][0]
            if delta_dis < 0:
                global_ht_pos[0, j] = -int(dis2idx[-delta_dis])
            else:
                global_ht_pos[0, j] = int(dis2idx[delta_dis])

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
                local_h_mapping[0, j, h['pos'][0] - h_dl[h_i]:h['pos'][1] - h_dl[h_i]] = 1.0 / h_num / (
                        h['pos'][1] - h['pos'][0])

            for t_i, t in enumerate(t_list):
                if t['sent_id'] not in evidence_sents:
                    continue
                local_t_mapping[0, j, t['pos'][0] - t_dl[t_i]:t['pos'][1] - t_dl[t_i]] = 1.0 / t_num / (
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
                delta_dis = (dis_h['pos'][0] - h_dl[dis_h_idx]) - (dis_t['pos'][0] - t_dl[dis_t_idx])
            else:
                delta_dis = 0

            if delta_dis < 0:
                local_ht_pos[0, j] = -int(dis2idx[-delta_dis])
            else:
                local_ht_pos[0, j] = int(dis2idx[delta_dis])

            r = label['r']
            relation_multi_label[0, j, r] = 1

            relation_mask[0, j] = 1
            relation_label[0, j] = r

            max_h_t_cnt = max_h_t_cnt + 1

        for j, triple in enumerate(ins['na_triple'], max_h_t_cnt):
            vertexSet = copy.deepcopy(ins['vertexSet'])

            evidence_sents = copy.deepcopy(triple['evidence'])
            words = []

            for sent in evidence_sents:
                words += ins['sents'][sent]

            max_local_wn = max(len(words), max_local_wn)

            for t, word in enumerate(words):
                word = word.lower()
                if t < self.MAX_LENGTH:
                    if word in self.word2id:
                        local_context_idxs[0, j, t] = self.word2id[word]
                    else:
                        local_context_idxs[0, j, t] = self.word2id['UNK']

            for t in range(t + 1, self.MAX_LENGTH):
                local_context_idxs[0, j, t] = self.word2id['BLANK']

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

                    local_context_pos[0, j, v['pos'][0] - dl:v['pos'][1] - dl] = pos_idx
                    local_context_ner[0, j, v['pos'][0] - dl:v['pos'][1] - dl] = self.ner2id[v['type']]

                if pos_flag:
                    pos_idx = pos_idx + 1
                    pos_flag = False

            h_list = vertexSet[triple['ht'][0]]
            t_list = vertexSet[triple['ht'][1]]

            for h_i, h in enumerate(h_list):
                global_h_mapping[0, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(h_list) / (h['pos'][1] - h['pos'][0])

            for t_i, t in enumerate(t_list):
                global_t_mapping[0, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(t_list) / (t['pos'][1] - t['pos'][0])

            delta_dis = h_list[0]['pos'][0] - t_list[0]['pos'][0]
            if delta_dis < 0:
                global_ht_pos[0, j] = -int(dis2idx[-delta_dis])
            else:
                global_ht_pos[0, j] = int(dis2idx[delta_dis])

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
                local_h_mapping[0, j, h['pos'][0] - h_dl[h_i]:h['pos'][1] - h_dl[h_i]] = 1.0 / h_num / (
                        h['pos'][1] - h['pos'][0])

            for t_i, t in enumerate(t_list):
                if t['sent_id'] not in evidence_sents:
                    continue
                local_t_mapping[0, j, t['pos'][0] - t_dl[t_i]:t['pos'][1] - t_dl[t_i]] = 1.0 / t_num / (
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
                delta_dis = (dis_h['pos'][0] - h_dl[dis_h_idx]) - (dis_t['pos'][0] - t_dl[dis_t_idx])
            else:
                delta_dis = 0

            if delta_dis < 0:
                local_ht_pos[0, j] = -int(dis2idx[-delta_dis])
            else:
                local_ht_pos[0, j] = int(dis2idx[delta_dis])

            r = 0
            relation_multi_label[0, j, r] = 1

            relation_mask[0, j] = 1
            relation_label[0, j] = r

            max_h_t_cnt = max_h_t_cnt + 1

        max_local_wn = max_global_wn

        input_data = {'global_context_idxs': global_context_idxs[:, :max_global_wn],
                      'global_context_pos': global_context_pos[:, :max_global_wn],
                      'global_context_ner': global_context_ner[:, :max_global_wn],
                      'global_h_mapping': global_h_mapping[:, :max_h_t_cnt, :max_global_wn],
                      'global_t_mapping': global_t_mapping[:, :max_h_t_cnt, :max_global_wn],
                      'global_ht_pos': global_ht_pos[:, :max_h_t_cnt],
                      'local_context_idxs': local_context_idxs[:, :max_h_t_cnt, :max_local_wn],
                      'local_context_pos': local_context_pos[:, :max_h_t_cnt, :max_local_wn],
                      'local_context_ner': local_context_ner[:, :max_h_t_cnt, :max_local_wn],
                      'local_h_mapping': local_h_mapping[:, :max_h_t_cnt, :max_local_wn],
                      'local_t_mapping': local_t_mapping[:, :max_h_t_cnt, :max_local_wn],
                      'local_ht_pos': local_ht_pos[:, :max_h_t_cnt],
                      'relation_label': relation_label[:, :max_h_t_cnt],
                      'relation_multi_label': relation_multi_label[:, :max_h_t_cnt, :],
                      'relation_mask': relation_mask[:, :max_h_t_cnt],
                      'max_global_wn': max_global_wn,
                      'max_local_wn': max_local_wn,
                      'max_h_t_cnt': max_h_t_cnt,
                      }

        return input_data