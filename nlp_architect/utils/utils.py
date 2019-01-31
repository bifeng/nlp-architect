from gensim.models import KeyedVectors
from tqdm import tqdm
import numpy as np


def save_embedding(word2vecfilename, term_index, embedfilename):
    embed = {}
    model = KeyedVectors.load_word2vec_format(word2vecfilename)

    embedfile = open(embedfilename, 'w')
    for key, value in tqdm(term_index.items()):
        try:
            embed[value] = model.wv[key]
        except:  # embedding OOV字符
            embed[value] = np.float32(np.random.uniform(-0.2, 0.2, 300))

        embedfile.write(str(value) + '\t' + " ".join(map(str, embed[value])))
        embedfile.write('\n')
    embedfile.close()
    print('[%s]\n\tEmbedding size: %d' % (term_index, len(embed)), end='\n')
    return


# Read Embedding File
def read_embedding(filename):
    embed = {}
    # for line in open(filename, encoding='utf-8'):
    #     line = line.strip().split()
    #     embed[line[0]] = list(map(float, line[1:]))
    for line in open(filename):
        line = line.strip().split()
        embed[int(line[0])] = list(map(float, line[1:]))
    print('[%s]\n\tEmbedding size: %d' % (filename, len(embed)), end='\n')
    return embed


def build_matrix(embed_dict, vocab_size, embed_size):
    _PAD_ = vocab_size
    embed_dict[_PAD_] = np.zeros((embed_size,), dtype=np.float32)
    embed = np.float32(np.random.uniform(-0.02, 0.02, [vocab_size, embed_size]))
    matrix = convert_embed_2_numpy(embed_dict, embed=embed)
    return matrix


# Convert Embedding Dict 2 numpy array
def convert_embed_2_numpy(embed_dict, max_size=0, embed=None):
    feat_size = len(embed_dict[list(embed_dict.keys())[0]])
    if embed is None:
        embed = np.zeros((max_size, feat_size), dtype=np.float32)

    if len(embed_dict) > len(embed):
        raise Exception("vocab_size %d is larger than embed_size %d, change the vocab_size in the config!"
                        % (len(embed_dict), len(embed)))

    # for index, k in enumerate(embed_dict):
    #     embed[index] = np.array(embed_dict[k])
    for k in embed_dict:
        embed[k-1] = np.array(embed_dict[k])
    print('Generate numpy embed:', str(embed.shape), end='\n')
    return embed

