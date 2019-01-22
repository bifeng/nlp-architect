from fuzzywuzzy import fuzz

# todo
# select similarity metrics more flexible


class StringModel(object):
    def __init__(self):
        pass

    def similarity(self,data, corpus):

        fuzzpartial_sim = []
        fuzztoken_set_sim = []
        for i in data:
            doc1 = corpus.get(i[0])
            doc2 = corpus.get(i[1])
            fuzzpartial_sim.append(fuzz.partial_ratio(doc1,doc2))
            fuzztoken_set_sim.append(fuzz.token_set_ratio(doc1,doc2))

        return {'fuzzpartial_sim':fuzzpartial_sim, 'fuzztoken_set_sim':fuzztoken_set_sim}
