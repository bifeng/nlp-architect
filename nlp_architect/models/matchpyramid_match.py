import os
import random
random.seed(49999)
import matchzoo as mz
import pandas as pd
from nlp_architect.data.preparation import *
from nlp_architect.utils.utils import *
from nlp_architect.config.path_config import modelsave_path, dataset_path


class MatchPyramid(object):
    def __init__(self, preprocessordir, embeddingdir,embeddingmatricdir,modeldir):
        self.preprocessordir = preprocessordir
        self.embeddingdir = embeddingdir
        self.embeddingmatricdir = embeddingmatricdir
        self.modeldir = modeldir

    @staticmethod
    def prepare_train(train_pack, valid_pack):
        # pre-process
        # 修改源码-以支持中文分词及停用词
        preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=20, fixed_length_right=20,
                                                          remove_stop_words=True)
        train_pack_processed = preprocessor.fit_transform(train_pack)
        # # online prediction - pickle this file on disk
        preprocessor.save(preprocessordir)

        valid_pack_processed = preprocessor.transform(valid_pack)

        return preprocessor, train_pack_processed, valid_pack_processed

    def prepare_test(self, test_pack):
        preprocessor = mz.load_preprocessor(self.preprocessordir)
        test_pack_processed = preprocessor.transform(test_pack)
        return test_pack_processed

    @staticmethod
    def model_build(preprocessor, train_pack_processed, valid_pack_processed):
        # model
        ranking_task = mz.tasks.Ranking(loss=mz.losses.RankHingeLoss())
        ranking_task.metrics = [
            mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
            mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
            mz.metrics.MeanAveragePrecision()
        ]

        model = mz.models.MatchPyramid()
        model.params['input_shapes'] = preprocessor.context['input_shapes']
        model.params['task'] = ranking_task
        model.params['embedding_input_dim'] = preprocessor.context['vocab_size']
        model.params['embedding_output_dim'] = 300
        model.params['embedding_trainable'] = True
        model.params['num_blocks'] = 1
        model.params['kernel_count'] = [64]
        model.params['kernel_size'] = [[3, 3]]
        model.params['dpool_size'] = [3, 10]
        model.params['optimizer'] = 'adam'
        model.params['dropout_rate'] = 0.1
        model.guess_and_fill_missing_params()
        model.build()
        model.compile()
        model.backend.summary()
        model.params.completed()
        print(model.params)

        # pre-train embedding
        save_embedding(embeddingdir, preprocessor.context['vocab_unit'].state['term_index'], embeddingmatricdir)

        embed_dict = read_embedding(filename=embeddingmatricdir)
        embedding_matrix = build_matrix(embed_dict, preprocessor.context['vocab_size'], embed_size=300)
        # # online prediction - load this file in memory
        model.load_embedding_matrix(embedding_matrix)

        # training
        train_generator = mz.DPoolPairDataGenerator(train_pack_processed,
                                                    fixed_length_left=20,
                                                    fixed_length_right=20,
                                                    num_dup=2,
                                                    num_neg=1,
                                                    batch_size=20)

        valid_generator = mz.DPoolDataGenerator(valid_pack_processed,
                                                fixed_length_left=20,
                                                fixed_length_right=20,
                                                batch_size=20)

        pred_x, pred_y = valid_generator[:]
        evaluate = mz.callbacks.EvaluateAllMetrics(model, x=pred_x, y=pred_y, batch_size=len(pred_y))
        history = model.fit_generator(train_generator, epochs=20, callbacks=[evaluate], verbose=2, workers=30,
                                      use_multiprocessing=True)
        model.save(modeldir)

    def predict(self,test_pack_processed):
        model = mz.load_model(self.modeldir)
        test_generator = mz.DPoolDataGenerator(test_pack_processed,
                                                fixed_length_left=20,
                                                fixed_length_right=20,
                                                batch_size=20)
        pred_x, pred_y = test_generator[:]
        predict_value = model.predict(pred_x, batch_size=len(pred_y))  # batch_size
        return predict_value


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 2"

    traindatadir = dataset_path + 'all_qq_qa_pairs_random4.txt'
    preprocessordir = modelsave_path + 'matchpyramidrank_model'
    embeddingdir = dataset_path + 'wiki_quesbank_quesauto_model.vec'
    embeddingmatricdir = dataset_path + 'embed_quesbank_quesauto_model'
    modeldir = modelsave_path + 'matchpyramidrank_model'

    prepare = Preparation()
    corpus, relations = prepare.run_with_one_corpus(traindatadir)
    prepare.save_corpus(dataset_path + 'corpus.txt', corpus)

    rel_train, rel_valid, rel_test = prepare.split_train_valid_test_for_ranking(relations, [0.8, 0.1, 0.1])
    prepare.save_relation(dataset_path + 'relation_train.txt', rel_train)
    prepare.save_relation(dataset_path + 'relation_valid.txt', rel_valid)
    prepare.save_relation(dataset_path + 'relation_test.txt', rel_test)
    print('preparation finished ...')

    # # build DataPack
    # "DataPack.pack"
    train = prepare.restore(corpus, rel_train)
    train_data = pd.DataFrame(train)
    train_data.rename(columns={'d1':'text_left','d2':'text_right'},inplace=True)
    train_data['label'] = train_data['label'].astype('int32')
    train_pack = mz.pack(train_data)
    # train_pack.frame().head()

    valid = prepare.restore(corpus, rel_valid)
    valid_data = pd.DataFrame(valid)
    valid_data.rename(columns={'d1':'text_left','d2':'text_right'},inplace=True)
    valid_data['label'] = valid_data['label'].astype('int32')
    valid_pack = mz.pack(valid_data)

    test = prepare.restore(corpus, rel_test)
    test_data = pd.DataFrame(test)
    test_data.rename(columns={'d1':'text_left','d2':'text_right'},inplace=True)
    test_data['label'] = test_data['label'].astype('int32')
    test_pack = mz.pack(test_data)

    matchmodel = MatchPyramid(preprocessordir, embeddingdir,embeddingmatricdir,modeldir)
    preprocessor, train_pack_processed, valid_pack_processed = matchmodel.prepare_train(train_pack, valid_pack)
    matchmodel.model_build(preprocessor, train_pack_processed, valid_pack_processed)

    test_pack_processed = matchmodel.prepare_test(test_pack)
    predict_value = matchmodel.predict(test_pack_processed)

