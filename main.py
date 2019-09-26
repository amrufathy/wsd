import random

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer

from data_loaders import SemcorDataLoader
from models import MultiTaskModel
from trainers import BlstmTrainer
from utils.args import get_args
from utils.config import process_config
from utils.dirs import create_dirs

random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)


def main():
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data generator.')
    semcor_dl = SemcorDataLoader(config)
    # X_trn, y_trn = semcor_dl.get_data()
    X_trn, y_trn = semcor_dl.read_data()
    # y_trn = y_trn[0]  # get senses only

    senses = semcor_dl.read_senses()
    poses = semcor_dl.get_pos()
    lexes = semcor_dl.get_lexes()

    # build vocab
    t_x = Tokenizer(oov_token='<UNK>')
    t_x.fit_on_texts(X_trn)

    input_vocab = list(t_x.word_index.keys())
    senses_vocab = input_vocab + senses

    # build output tokenizer
    t_y_senses = create_output_vocab_dict(senses_vocab)
    t_y_pos = create_output_vocab_dict(poses, start=2)
    t_y_lex = create_output_vocab_dict(lexes, start=2)

    mask = np.asarray(semcor_dl.create_mask(t_y_senses.word_index, len(X_trn)))

    # save tokenizer
    with open('tokenizer.pic', 'wb') as f:
        import pickle
        pickle.dump(t_y_senses, f)

    # set config params
    config.model.vocab_size = len(input_vocab)
    config.model.output_size = config.model.vocab_size + len(senses) + 1
    config.model.pos_size = len(poses) + 2
    config.model.lex_size = len(lexes) + 2
    config.trainer.examples_size = len(X_trn)

    print('Create the model.')
    model = MultiTaskModel(config).model
    model.summary()

    print('Create the trainer')
    trainer = BlstmTrainer(model, config)

    print('Start training the model.')
    # convert to sequences
    use_elmo = bool(config.model.use_elmo)

    if not use_elmo:
        X_trn = t_x.texts_to_sequences(X_trn)

    X_trn = np.asarray(X_trn)
    y_trn = [np.asarray(t_y_senses.texts_to_sequences(y_trn[0])),
             np.asarray(t_y_pos.texts_to_sequences(y_trn[1])),
             np.asarray(t_y_lex.texts_to_sequences(y_trn[2]))]

    trainer.train((X_trn, y_trn, mask))

    model.save('model.h5')


def create_output_vocab_dict(vocab, start=0):
    t = Tokenizer(oov_token='<UNK>')
    t.word_index = (dict((w, i) for i, w in enumerate(vocab, start)))
    t.word_index.update({'<UNK>': 1})
    t.index_word = dict((i, w) for w, i in t.word_index.items())

    return t


if __name__ == '__main__':
    main()
