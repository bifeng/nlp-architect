import hashlib
import pandas as pd

# transform docs to ids more efficiency!
# refer:
# matchzoo 1.0



DEBUG_MODE = True


class Preparation(object):
    def __init__(self, path):
        self.path = path

    def _get_text_id(self, hashid, text, idtag='T'):
        hash_obj = hashlib.sha1(text.encode('utf8'))  # if the text are the same, then the hash_code are also the same
        hex_dig = hash_obj.hexdigest()
        if hex_dig in hashid:
            return hashid[hex_dig]
        else:
            tid = idtag + str(len(hashid))  # start from 0, 1, 2, ...
            hashid[hex_dig] = tid
            return tid

    def load_data(self):
        '''
        :return:
        corpus - dict key:id,value:doc
        '''
        df = pd.read_excel(self.path)
        df.rename(columns={'question':'doc1', 'top_question':'doc2'},inplace=True)
        df = df[['doc1', 'doc2']]

        if DEBUG_MODE:
            df = df.head(500)

        # clear datasets
        qa_doc = self.clear(df)


        hashid = {}
        corpus = {}
        docpair = []
        for index, row in qa_doc.iterrows():
            id1 = self._get_text_id(hashid, row['doc1'])  # dict as mutable parameters
            id2 = self._get_text_id(hashid, row['doc2'])
            corpus[id1] = row['doc1']
            corpus[id2] = row['doc2']
            docpair.append((id1,id2))

        return docpair, corpus

    def clear(self, data):
        # todo clear :empty or too short answer
        data = data[data.doc1.notnull()]
        data = data[data.doc2.notnull()]
        return data


