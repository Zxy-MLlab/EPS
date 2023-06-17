import re
import json
import os
import numpy as np
import tensorflow as tf
import ClassModel
import PolicyModel
import train_cl_GL
import train_rl_GL
import Process
import select_evidence

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

IN_PATH = 'prepro_data'
OUT_PATH = 'prepro_data'
MODEL_PATH = 'output'
MAX_LENGTH = 512
CLASS_NUM = 97
CL_LEARNING_RATE = 1e-3
RL_LEARNING_RATE = 1e-5
HIS_MAX_LENGTH = 512
CUR_MAX_LENGTH = 64
H_T_LIMIT = 1800
TRAIN_EPOCHES = 1


def main():
    reader = Reader(in_path=IN_PATH, out_path=OUT_PATH)
    rel2id, id2rel = reader.read_relations()
    word2id, ner2id, char2id = reader.read_word_id()

    processer = Process.Process(MAX_LENGTH=MAX_LENGTH, HIS_MAX_LENGTH=HIS_MAX_LENGTH, CUR_MAX_LENGTH=CUR_MAX_LENGTH, H_T_LIMIT=H_T_LIMIT,
                                CLASS_NUM=CLASS_NUM, word2id=word2id, ner2id=ner2id)

    g_cmodel = tf.Graph()
    sess1 = tf.Session(graph=g_cmodel)

    with g_cmodel.as_default():
        with sess1.as_default():
            CModel = ClassModel.ClassModel(sess=sess1, model_path=MODEL_PATH, reader=reader, MAX_LENGTH=MAX_LENGTH,
                                           CLASS_NUM=CLASS_NUM,
                                           model_flag=False)
            # CModel.restore('output/ClassModel/cmodel.ckpt')

    g_cmodel.finalize()

    g_pmodel = tf.Graph()
    sess2 = tf.Session(graph=g_pmodel)

    with g_pmodel.as_default():
        with sess2.as_default():
            PModel = PolicyModel.PolicyModel(sess=sess2, reader=reader, MAX_LENGTH=MAX_LENGTH, RL_LEARNING_RATE=RL_LEARNING_RATE)
            # PModel.restore('output/PolicyModel/pmodel.ckpt')
    g_pmodel.finalize()


    # train ClassModel
    print("start train ClassModel...")
    train_cl_GL.main(DATH_PATH='prepro_data', CModel=CModel, id2rel=id2rel, word2id=word2id, ner2id=ner2id,
                     MODEL_PATH='output', TRAIN_EPOCHES=5, TEST_EPOCH=1, CL_LEARNING_RATE=1e-3)
    print("finish train ClassModel!")

    # train PolicyModel
    print("start train PolicyModel...")
    train_rl_GL.train_rl(processer=processer, CModel=CModel, PModel=PModel, MODEL_PATH='output', TRAIN_EPOCHES=100)
    print("finish train PolicyModel!")

    # select evidence sents by current policy
    print("start select evidence sents...")
    select_evidence.main(processer=processer, CModel=CModel, PModel=PModel)
    print("finish select evidence sents!")

    # union train ClassModel and PolicyModel
    print("start union train...")

    for epoch in range(TRAIN_EPOCHES):
        print("union train epoch: %s"%str(epoch + 1))

        print("start train ClassModel...")
        train_cl_GL.main(DATH_PATH='union_data', CModel=CModel, id2rel=id2rel, word2id=word2id, ner2id=ner2id,
                         MODEL_PATH='union_model', TRAIN_EPOCHES=25, TEST_EPOCH=1, CL_LEARNING_RATE=1e-5)
        print("finish train ClassModel!")

        if epoch == TRAIN_EPOCHES-1:
            break

        print("start train PolicyModel...")
        train_rl_GL.train_rl(processer=processer, CModel=CModel, PModel=PModel, MODEL_PATH='union_model', TRAIN_EPOCHES=100)
        print("finish train PolicyModel!")

        if os.path.exists('union_data/dev_train.json') or os.path.exists('union_data/dev_dev.json'):
            os.remove('union_data/dev_train.json')
            os.remove('union_data/dev_dev.json')

        print("start select evidence sents...")
        select_evidence.main(processer=processer, CModel=CModel, PModel=PModel)
        print("finish select evidence sents!")

    print("finish union train!")

    return

if __name__ == '__main__':
    main()