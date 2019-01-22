import pandas as pd


class ChatData(object):
    def __init__(self, path):
        self.path = path

    def load_data(self):
        '''
        :return:
        '''
        # todo
        # 1. doc doc-id map - matrix
        # 2. save matrix and return ids
        df = pd.read_excel(self.path)

        qa_doc = df

        qa_docall = [list(zip(qa_doc.index.tolist(),qa_doc[column].tolist())) for column in ['question','top_answer','top_question']]
        qa_doc_uni = list(set([leaf for branch in qa_docall for leaf in branch]))

        return qa_doc, qa_doc_uni

    def clear(self):
        # todo clear :empty or too short answer

        pass



# if __name__ == "__main__":
#     from nlp_architect.config.path_config import chatdatapath
#     datasets = ChatData(chatdatapath)
#     docs = datasets.load_data()
#     print(docs.head())
