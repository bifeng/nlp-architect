import jieba


class Preprocessor(object):
    # todo
    # 1. word-id embedding map - matrix
    # 2. save matrix and return ids
    def __init__(self):
        pass

    def tokenize(self, docs):
        '''
        :param docs: list or string
        :return:
        '''
        if isinstance(docs,list):
            return [list(jieba.cut(doc)) for doc in docs]
        else:
            return list(jieba.cut(docs))


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

