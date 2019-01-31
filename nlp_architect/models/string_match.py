from fuzzywuzzy import fuzz

# todo
# select similarity metrics more flexible


class StringModel(object):
    def __init__(self, data, corpus):
        '''
        :param data:
        :param corpus:
        '''
        self.data = data
        self.corpus = corpus
        pass

    def fuzzpartial(self):

        fuzzpartial_sim = []
        for i in self.data:
            doc1 = self.corpus.get(i[0])
            doc2 = self.corpus.get(i[1])
            fuzzpartial_sim.append(fuzz.partial_ratio(doc1,doc2))

        return fuzzpartial_sim

    def fuzztokenset(self):

        fuzztoken_set_sim = []
        for i in self.data:
            doc1 = self.corpus.get(i[0])
            doc2 = self.corpus.get(i[1])
            fuzztoken_set_sim.append(fuzz.token_set_ratio(doc1,doc2))

        return fuzztoken_set_sim



# if __name__ == "__main__":
#     from nlp_architect.data.chat_datasets import Preparation
#     from nlp_architect.config.path_config import chatdatapath
#     datasets = Preparation(chatdatapath)
#     docpair, corpus = datasets.load_data()
#
#     stringm = StringModel()
#     features = stringm.fuzzpartial(docpair, corpus)
#     print(features)
