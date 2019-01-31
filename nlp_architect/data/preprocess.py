# todo
# 1. word-id embedding map - matrix
# 2. save matrix and return ids

import jieba
import codecs
from tqdm import tqdm
from nlp_architect.models.sentence_vector import sentence_vector


class Preprocess(object):
    def __init__(self,
                 word_seg_config = {},
                 word_lower_config = {},
                 ):
        # set default configuration
        self._word_seg_config = { 'enable': True}
        self._word_lower_config = { 'enable': True }

        self._word_seg_config.update(word_seg_config)
        self._word_lower_config.update(word_lower_config)

    def run(self, file_path):
        print('load...')
        dids, docs = Preprocess.load(file_path)

        if self._word_seg_config['enable']:
            print('word_seg...')
            docs = Preprocess.word_seg(docs)

        if self._word_lower_config['enable']:
            print('word_lower...')
            docs = Preprocess.word_lower(docs)

        return dids, docs

    @staticmethod
    def parse(line):
        subs = line.split(' ', 1)
        if 1 == len(subs):
            return subs[0], ''
        else:
            return subs[0], subs[1]

    @staticmethod
    def load(file_path):
        dids = list()
        docs = list()
        f = codecs.open(file_path, 'r', encoding='utf8')
        for line in tqdm(f):
            line = line.strip()
            if '' != line:
                did, doc = Preprocess.parse(line)
                dids.append(did)
                docs.append(doc)
        f.close()
        return dids, docs

    @staticmethod
    def word_seg(docs):
        docs = [list(jieba.cut(sent)) for sent in docs]
        return docs

    @staticmethod
    def word_lower(docs):
        docs = [[w.lower() for w in ws] for ws in tqdm(docs)]
        return docs

    def corpus2vector(self, docs, model):
        corpus_vector = []
        for doc in docs:
            corpus_vector.append(sentence_vector(doc,model))
        return corpus_vector

# if __name__ == "__main__":
#     from nlp_architect.data.chat_datasets import Preparation
#     from nlp_architect.config.path_config import chatdatapath
#     datasets = Preparation(chatdatapath)
#     __, corpus = datasets.load_data()
#
#     doc_list = list(corpus.values())
#     preprocess = Preprocessor()
#     doc_seg = preprocess.tokenize(doc_list)
#     print(doc_seg[:3])

