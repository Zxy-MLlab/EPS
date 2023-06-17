import numpy as np
import os
import json
from nltk.tokenize import WordPunctTokenizer
import argparse
from collections import defaultdict
parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type = str, default =  "../../data")
parser.add_argument('--out_path', type = str, default = "rl_data")

args = parser.parse_args()
in_path = args.in_path
out_path = args.out_path
case_sensitive = False

char_limit = 16
train_distant_file_name = os.path.join(in_path, 'train_distant.json')
train_annotated_file_name = os.path.join(in_path, 'train_annotated.json')
dev_file_name = os.path.join(in_path, 'dev.json')
test_file_name = os.path.join(in_path, 'test.json')

rel2id = json.load(open(os.path.join(out_path, 'rel2id.json'), "r"))
id2rel = {v:u for u,v in rel2id.items()}
json.dump(id2rel, open(os.path.join(out_path, 'id2rel.json'), "w"))
fact_in_train = set([])
fact_in_dev_train = set([])



def extract_path(data, keep_sent_order):
    sents = data["sents"]
    nodes = [[] for _ in range(len(data['sents']))]
    e2e_sent = defaultdict(dict)

    # create mention's list for each sentence
    # 存储每个句子中有哪些实体提及：nodes[[],[],[]...]
    for ns_no, ns in enumerate(data['vertexSet']):
        for n in ns:
            sent_id = int(n['sent_id'])
            nodes[sent_id].append(ns_no)

    for sent_id in range(len(sents)):
        for n1 in nodes[sent_id]:
            for n2 in nodes[sent_id]:
                if n1 == n2:
                    continue
                if n2 not in e2e_sent[n1]:
                    e2e_sent[n1][n2] = set()
                e2e_sent[n1][n2].add(sent_id)

    # 2-hop Path
    path_two = defaultdict(dict)
    entityNum = len(data['vertexSet'])
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n1 == n2:
                continue
            for n3 in range(entityNum):
                if n3 == n1 or n3 == n2:
                    continue
                if not (n3 in e2e_sent[n1] and n2 in e2e_sent[n3]):
                    continue
                for s1 in e2e_sent[n1][n3]:
                    for s2 in e2e_sent[n3][n2]:
                        if s1 == s2:
                            continue
                        if n2 not in path_two[n1]:
                            path_two[n1][n2] = []
                        cand_sents = [s1, s2]
                        if keep_sent_order == True:
                            cand_sents.sort()
                        path_two[n1][n2].append((cand_sents, n3))

    # 3-hop Path
    path_three = defaultdict(dict)
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n1 == n2:
                continue
            for n3 in range(entityNum):
                if n3 == n1 or n3 == n2:
                    continue
                if n3 in e2e_sent[n1] and n2 in path_two[n3]:
                    for cand1 in e2e_sent[n1][n3]:
                        for cand2 in path_two[n3][n2]:
                            if cand1 in cand2[0]:
                                continue
                            if cand2[1] == n1:
                                continue
                            if n2 not in path_three[n1]:
                                path_three[n1][n2] = []
                            cand_sents = [cand1] + cand2[0]
                            if keep_sent_order:
                                cand_sents.sort()
                            path_three[n1][n2].append((cand_sents, [n3, cand2[1]]))

    # Consecutive Path
    consecutive = defaultdict(dict)
    for h in range(entityNum):
        for t in range(h + 1, entityNum):
            for n1 in data['vertexSet'][h]:
                for n2 in data['vertexSet'][t]:
                    gap = abs(n1['sent_id'] - n2['sent_id'])
                    if gap > 2:
                        continue
                    if t not in consecutive[h]:
                        consecutive[h][t] = []
                        consecutive[t][h] = []
                    if n1['sent_id'] < n2['sent_id']:
                        beg, end = n1['sent_id'], n2['sent_id']
                    else:
                        beg, end = n2['sent_id'], n1['sent_id']

                    consecutive[h][t].append([[i for i in range(beg, end + 1)]])
                    consecutive[t][h].append([[i for i in range(beg, end + 1)]])

    # Merge
    merge = defaultdict(dict)
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n2 in path_two[n1]:
                merge[n1][n2] = path_two[n1][n2]
            if n2 in path_three[n1]:
                if n2 in merge[n1]:
                    merge[n1][n2] += path_three[n1][n2]
                else:
                    merge[n1][n2] = path_three[n1][n2]

            if n2 in consecutive[n1]:
                if n2 in merge[n1]:
                    merge[n1][n2] += consecutive[n1][n2]
                else:
                    merge[n1][n2] = consecutive[n1][n2]

    # Remove redundency
    tp_set = set()
    for n1 in merge.keys():
        for n2 in merge[n1].keys():
            hash_set = set()
            new_list = []
            for t in merge[n1][n2]:
                if tuple(t[0]) not in hash_set:
                    hash_set.add(tuple(t[0]))
                    new_list.extend(t[0])
            new_list = list(set(new_list))
            if keep_sent_order:
                new_list.sort()
            merge[n1][n2] = new_list

    return merge

def init(data_file_name, rel2id, max_length = 512, is_training = True, suffix=''):
    ori_data = json.load(open(data_file_name))

    data = []

    for i in range(len(ori_data)):
        Ls = [0]
        L = 0
        LS = []

        for x in ori_data[i]['sents']:
            L += len(x)
            Ls.append(L)
            LS.append(len(x))

        all_sents = list(range(len(ori_data[i]['sents'])))

        vertexSet = ori_data[i]['vertexSet']
        for j in range(len(vertexSet)):
            for k in range(len(vertexSet[j])):
                vertexSet[j][k]['sent_id'] = int(vertexSet[j][k]['sent_id'])

                sent_id = vertexSet[j][k]['sent_id']
                dl = Ls[sent_id]
                pos1 = vertexSet[j][k]['pos'][0]
                pos2 = vertexSet[j][k]['pos'][1]
                vertexSet[j][k]['pos'] = (pos1 + dl, pos2 + dl)
                if pos1 + dl > max_length:
                    print(pos1)
        merge_path = extract_path(data=ori_data[i], keep_sent_order=True)

        ori_data[i]['vertexSet'] = vertexSet

        item = {}
        item['vertexSet'] = vertexSet
        item['Ls'] = LS
        labels = ori_data[i].get('labels', [])

        vertexSet = ori_data[i]['vertexSet']

        train_triple = set([])
        new_labels = []
        for label in labels:
            evidence_sents = []

            h = label['h']
            t = label['t']

            if t in merge_path[h]:
                evidence_sents = merge_path[h][t]
                label['e'] = True
            else:
                for h_m in vertexSet[h]:
                    sent_id = int(h_m['sent_id'])
                    if sent_id not in evidence_sents:
                        evidence_sents.append(sent_id)
                        break

                for t_m in vertexSet[t]:
                    sent_id = int(t_m['sent_id'])
                    if sent_id not in evidence_sents:
                        evidence_sents.append(sent_id)
                        break

                label['e'] = False

            evidence_sents.sort()
            label['evidence'] = evidence_sents

            rel = label['r']
            assert (rel in rel2id)
            label['r'] = rel2id[label['r']]

            train_triple.add((label['h'], label['t']))

            if suffix == '_train':
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_dev_train.add((n1['name'], n2['name'], rel))

            if is_training:
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_train.add((n1['name'], n2['name'], rel))

            else:
                # fix a bug here
                label['intrain'] = False
                label['indev_train'] = False

                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        if (n1['name'], n2['name'], rel) in fact_in_train:
                            label['intrain'] = True

                        if suffix == '_dev' or suffix == '_test':
                            if (n1['name'], n2['name'], rel) in fact_in_dev_train:
                                label['indev_train'] = True

            new_labels.append(label)

        item['labels'] = new_labels
        item['title'] = ori_data[i]['title']

        na_triple = []
        for j in range(len(vertexSet)):
            for k in range(len(vertexSet)):
                if (j != k):
                    if (j, k) not in train_triple:
                        triple = {}
                        triple['ht'] = (j,k)

                        evidence_sents = []

                        if k in merge_path[j]:
                            evidence_sents = merge_path[j][k]
                            triple['e'] = True
                        else:
                            for h_m in vertexSet[j]:
                                sent_id = int(h_m['sent_id'])
                                if sent_id not in evidence_sents:
                                    evidence_sents.append(sent_id)
                                    break

                            for t_m in vertexSet[k]:
                                sent_id = int(t_m['sent_id'])
                                if sent_id not in evidence_sents:
                                    evidence_sents.append(sent_id)
                                    break

                            triple['e'] = False

                        evidence_sents.sort()

                        triple['evidence'] = evidence_sents

                        na_triple.append(triple)

        item['na_triple'] = na_triple
        item['sents'] = ori_data[i]['sents']

        all_triples = []
        for label in labels:
            all_triples.append(label)

        for triple in na_triple:
            label = {}
            label['h'] = triple['ht'][0]
            label['t'] = triple['ht'][1]
            label['r'] = 0
            label['evidence'] = triple['evidence']
            label['e'] = triple['e']
            all_triples.append(label)

        item['all_triples'] = all_triples

        data.append(item)

    print("Saving files")
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"

    json.dump(data, open(os.path.join(out_path, name_prefix + suffix + '.json'), "w"))

    return

# init(train_distant_file_name, rel2id, max_length = 512, is_training = True, suffix='')
init(train_annotated_file_name, rel2id, max_length = 512, is_training = False, suffix='_train')
init(dev_file_name, rel2id, max_length = 512, is_training = False, suffix='_dev')
init(test_file_name, rel2id, max_length = 512, is_training = False, suffix='_test')