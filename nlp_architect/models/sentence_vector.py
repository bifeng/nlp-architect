import numpy as np


def sentence_vector(doc, model, dim=300):
    """
    :param doc: segment doc, list type
    :param model: word2vec
    :return:
    """
    vector_ = np.zeros((dim,))
    for word in doc:
        try:
            vector_ += model.wv[word]
        except:
            vector_ += 0

    return vector_

