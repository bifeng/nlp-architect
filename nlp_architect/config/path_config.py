import os
from pathlib import Path

root_path = str(Path(os.path.abspath(__file__)).parent.parent.parent)
dataset_path = root_path + '/datasets/'
modelsave_path = root_path + '/nlp_architect/data/model_save/'


chatdatapath = dataset_path + 'chat_test/all_qq_qa_pairs.xlsx'
tfidfmodelpath = modelsave_path + 'tfidfmodel'

