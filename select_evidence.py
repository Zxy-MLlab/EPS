import re
import os
import json
import numpy as np
import copy
import tqdm

IN_PATH = 'rl_data'
OUT_PATH = 'rl_data'

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

def select(origin_data, processer, PModel, CModel):
    select_data = []
    for i,_ in enumerate(tqdm.tqdm(origin_data)):
        data = copy.deepcopy(origin_data[i])
        labels = copy.deepcopy(data['labels'])
        all_triples = copy.deepcopy(data['all_triples'])
        all_sents = list(range(len(data['sents'])))
        ins = copy.deepcopy(origin_data[i])

        input_data = processer.process_cldata(ins=data)
        logits, _ = CModel.get_rewards(input_data)

        select_order = []
        for x in range(len(all_triples)):
            if (all_triples[x]['e'] == False) & (np.max(logits[0][x]) < 0.85):
                select_order.append(x)

        if len(select_order) == 0:
            select_data.append(ins)
            continue

        for s_o in select_order:
            label = copy.deepcopy(all_triples[s_o])
            rl_evidence_sents = label['evidence']

            for s_i in all_sents:
                if s_i in rl_evidence_sents:
                    continue

                context_idxs, context_pos, context_ner, \
                h_mapping, t_mapping = processer.process_his_data(data, label, rl_evidence_sents)

                sen_idx, sen_pos, sen_ner = processer.process_cur_data(data, s_i)

                action,_ = PModel.get_action(context_idxs=context_idxs, context_pos=context_pos, context_ner=context_ner,
                                             h_mapping=h_mapping, t_mapping=t_mapping,
                                             sen_idx=sen_idx, sen_pos=sen_pos, sen_ner=sen_ner)

                if action == 1:
                    rl_evidence_sents.append(s_i)

            rl_evidence_sents.sort()
            if s_o < len(labels):
                ins['labels'][s_o]['evidence'] = rl_evidence_sents
            else:
                ins['na_triple'][s_o - len(labels)]['evidence'] = rl_evidence_sents

        select_data.append(ins)

    return select_data

def save(prefix, data):
    path = 'union_data' + '/' + prefix + '.json'
    json.dump(data, open(path, 'w'))

    return


def main(processer, CModel, PModel):
    reader = Reader(in_path=IN_PATH, out_path=OUT_PATH)

    print("select train data ...")
    prefix = 'dev_train'
    train_data = reader.read_origin_data(prefix)

    select_train_data = select(train_data, processer, PModel, CModel)
    save(prefix, select_train_data)

    print("select dev data ...")
    prefix = 'dev_dev'
    dev_data = reader.read_origin_data(prefix)

    select_dev_data = select(dev_data, processer, PModel, CModel)
    save(prefix, select_dev_data)

    print("select test data ...")
    prefix = 'dev_test'
    test_data = reader.read_origin_data(prefix)

    select_test_data = select(test_data, processer, PModel, CModel)
    save(prefix, select_test_data)

    return