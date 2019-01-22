import time
import codecs
import pickle
from gensim import corpora, models, similarities


DEBUG_MODE = False


class TfidfModel(object):
    # todo fit/transform method
    def __init__(self, corpus_file, word2id):
        size = 10000000
        with codecs.open(corpus_file, "r", "utf-8") as rfd:
            data = [s.strip().split("\t") for s in rfd.readlines()[:size]]
            self.contexts = [[w for w in s.split() if w in word2id] for s, _ in data]
            self.responses = [s.replace(" ", "") for _, s in data]

        self.corpus_mm = self.tfidf_model[self.corpus]

    def fit(self, min_freq=1):
        # Create tfidf model.
        self.dct = corpora.Dictionary(self.contexts)
        # Filter low frequency words from dictionary.
        low_freq_ids = [id_ for id_, freq in
                        self.dct.dfs.items() if freq <= min_freq]
        self.dct.filter_tokens(low_freq_ids)
        self.dct.compactify()
        # Build tfidf-model.
        self.corpus = [self.dct.doc2bow(s) for s in self.contexts]
        self.tfidf_model = models.TfidfModel(self.corpus)


    def _text2vec(self, text):
        bow = self.dct.doc2bow(text)
        return self.tfidf_model[bow]

    def similarity(self, query, size=10):
        vec = self._text2vec(query)
        sims = self.index[vec]
        sim_sort = sorted(list(enumerate(sims)),
                          key=lambda item: item[1], reverse=True)
        return sim_sort[:size]

    def get_docs(self, sim_items):
        docs = [self.contexts[id_] for id_, score in sim_items]
        answers = [self.responses[id_] for id_, score in sim_items]
        return docs, answers




