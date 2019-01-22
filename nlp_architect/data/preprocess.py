import jieba


class Preprocessor(object):
    # todo
    # 1. word-id embedding map - matrix
    # 2. save matrix and return ids
    def __init__(self, docs):
        self.docs = docs

    def tokenize(self):
        return [list(jieba.cut(doc)) for doc in self.docs]





if __name__ == "__main__":
    from nlp_architect.data.chat_datasets import ChatData
    from nlp_architect.config.path_config import chatdatapath
    datasets = ChatData(chatdatapath)
    docs = datasets.load_data()

    print(docs.head())