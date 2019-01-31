from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis


class Word2vecModel(object):
    def __init__(self, data, corpus, model):
        '''
        :param data:
        :param corpus: segment doc, dict type
        :param model: word2vec
        '''
        self.data = data
        self.corpus = corpus
        self.model = model

    def load_corpus2vector(self):
        # todo
        return corpus_vector

    def wmd_(self):
        wmdistance = []
        for i in self.data:
            doc1 = self.corpus.get(i[0])
            doc2 = self.corpus.get(i[1])
            wmdistance.append(self.model.wmdistance(doc1, doc2))

        return wmdistance

    def cosine_(self):
        corpus_vector = self.load_corpus2vector()

        cosine_sim = []
        for i in self.data:
            doc1 = corpus_vector.get(i[0])
            doc2 = corpus_vector.get(i[1])
            cosine_sim.append(cosine(doc1,doc2))

        return cosine_sim

    def jaccard_(self):
        corpus_vector = self.load_corpus2vector()

        jaccard_sim = []
        for i in self.data:
            doc1 = corpus_vector.get(i[0])
            doc2 = corpus_vector.get(i[1])
            jaccard_sim.append(jaccard(doc1, doc2))

        return jaccard_sim


# if __name__ == "__main__":
#
#     from nlp_architect.data.chat_datasets import Preparation
#     from nlp_architect.data.preprocess import Preprocessor
#     from nlp_architect.config.path_config import chatdatapath
#     datasets = Preparation(chatdatapath)
#     docpair, corpus = datasets.load_data()
#
#     corpus_process = {}
#     preprocess = Preprocessor()
#     for key,value in corpus.items():
#         corpus_process[key] = preprocess.tokenize(value)


