import os
from pathlib import Path

root_path = str(Path(os.path.abspath(__file__)).parent.parent.parent)
dataset_path = root_path + '/datasets/chat_test/'
modelsave_path = root_path + '/nlp_architect/data/model_save/'


