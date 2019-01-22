import joblib
from scipy.spatial import distance
from nlp_architect.config.path_config import tfidfmodelpath
from nlp_architect.data.preprocess import Preprocessor
from sklearn.feature_extraction.text import TfidfVectorizer

# todo
# similarity calculate more efficiency!


preprocess = Preprocessor()
tokenize = preprocess.tokenize


class TfidfModel(object):
    def __init__(self):
        '''
        corpus: id doc
        '''
        self.vocabulary = None
        self.tokenizer = tokenize

    def fit(self, corpus):
        vectorizer = TfidfVectorizer(tokenizer=tokenize)
        docs = list(corpus.values())
        vectorizer.fit(docs)
        joblib.dump(vectorizer,tfidfmodelpath)
        X = vectorizer.transform(docs)
        return X

    def transform(self, newcorpus):
        vectorizer = joblib.load(tfidfmodelpath)
        X = vectorizer.transform(newcorpus)
        return X

    def similarity(self, data, corpus):
        X = self.fit(corpus)
        keyindex_map = {key:index for index,key in enumerate(corpus.keys())}

        sim_ = []
        for i in data:
            index1 = keyindex_map.get(i[0])
            index2 = keyindex_map.get(i[1])
            disvalue = distance.cosine(X[index1].toarray(), X[index2].toarray())
            sim_.append(disvalue)

        return {'tfidf_sim':sim_}


# if __name__ == "__main__":
#     from nlp_architect.data.chat_datasets import Preparation
#     from nlp_architect.config.path_config import chatdatapath
#     datasets = Preparation(chatdatapath)
#     docpair, corpus = datasets.load_data()
#
#     tfidf = TfidfModel()
#     features = tfidf.similarity(docpair, corpus)
#     print(features)
