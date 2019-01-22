import os
from pathlib import Path

root_path = str(Path(os.path.abspath(__file__)).parent.parent.parent)

chatdatapath = root_path + '/datasets/chat_test/all_qq_qa_pairs.xlsx'
