import codecs
import hashlib
import random

DEBUG_MODE = True


class Preparation(object):

    def __init__(self):
        pass

    def get_text_id(self, hashid, text, idtag='T'):
        hash_obj = hashlib.sha1(text.encode('utf8'))  # if the text are the same, then the hash_code are also the same
        hex_dig = hash_obj.hexdigest()
        if hex_dig in hashid:
            return hashid[hex_dig]
        else:
            tid = idtag + str(len(hashid))  # start from 0, 1, 2, ...
            hashid[hex_dig] = tid
            return tid

    def parse_line(self, line, delimiter='\t'):
        subs = line.split(delimiter)
        # print('line: \"{}\"'.format(line))
        # print('subs: ', len(subs))
        if 3 != len(subs):
            raise ValueError('format of data file wrong, should be \'label,text1,text2\'.')
        else:
            return subs[0], subs[1], subs[2]

    def run_with_one_corpus(self, file_path, encoding='utf8'):
        hashid = {}
        corpus = {}
        rels = []
        f = codecs.open(file_path, 'r', encoding=encoding)
        for index, line in enumerate(f):

            if DEBUG_MODE and index > 500:
                break

            line = line
            line = line.strip()
            label, t1, t2 = self.parse_line(line)
            id1 = self.get_text_id(hashid, t1, 'T')
            id2 = self.get_text_id(hashid, t2, 'T')
            corpus[id1] = t1
            corpus[id2] = t2
            rels.append((label, id1, id2))
        f.close()
        return corpus, rels


    @staticmethod
    def split_train_valid_test(relations, ratio=(0.8, 0.1, 0.1)):
        random.shuffle(relations)
        total_rel = len(relations)
        num_train = int(total_rel * ratio[0])
        num_valid = int(total_rel * ratio[1])
        valid_end = num_train + num_valid
        rel_train = relations[: num_train]
        rel_valid = relations[num_train: valid_end]
        rel_test = relations[valid_end:]
        return rel_train, rel_valid, rel_test

    @staticmethod
    def split_train_valid_test_for_ranking(relations, ratio=(0.8, 0.1, 0.1)):
        qid_group = set()
        for r, q, d in relations:
            qid_group.add(q)
        qid_group = list(qid_group)

        random.shuffle(qid_group)
        total_rel = len(qid_group)
        num_train = int(total_rel * ratio[0])
        num_valid = int(total_rel * ratio[1])
        valid_end = num_train + num_valid

        qid_train = qid_group[: num_train]
        qid_valid = qid_group[num_train: valid_end]
        qid_test = qid_group[valid_end:]

        def select_rel_by_qids(qids):
            rels = []
            qids = set(qids)
            for r, q, d in relations:
                if q in qids:
                    rels.append((r, q, d))
            return rels

        rel_train = select_rel_by_qids(qid_train)
        rel_valid = select_rel_by_qids(qid_valid)
        rel_test = select_rel_by_qids(qid_test)

        return rel_train, rel_valid, rel_test

    @staticmethod
    def restore(corpus, relations):
        docpairs = []
        for each in relations:
            temp = {}
            temp['label'] = each[0]
            temp['d1'] = corpus.get(each[1])
            temp['d2'] = corpus.get(each[2])
            docpairs.append(temp)
        return docpairs

    @staticmethod
    def save_corpus(file_path, corpus):
        f = codecs.open(file_path, 'w', encoding='utf8')
        for qid, text in corpus.items():
            f.write('%s %s\n' % (qid, text))
        f.close()


    @staticmethod
    def save_relation(file_path, relations):
        f = open(file_path, 'w')
        for rel in relations:
            f.write('%s %s %s\n' % (rel))
        f.close()





# if __name__ == "__main__":
#     from nlp_architect.config.path_config import chatdatapath
#     datasets = Preparation()
#     docpairs, corpus = datasets.run_with_one_corpus(chatdatapath)

