from os.path import join

import numpy as np
import tensorflow as tf

from data_loaders import SemevalDataLoader
from models import MFS
from models.attention import Attention
from models.elmo_layer import ElmoEmbeddingLayer
from utils.config import process_config
from utils.conversion import *


def predict_babelnet(input_path: str, output_path: str, resources_path: str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    config = process_config('configs/blstm_config.json')
    config.data.resources_path = resources_path
    # load model
    model: tf.keras.models.Model = tf.keras.models.load_model(join(resources_path, 'model.h5'), custom_objects={
        'ElmoEmbeddingLayer': ElmoEmbeddingLayer,
        'Attention': Attention
    })

    with open(join(resources_path, 'tokenizer.pic'), 'rb') as f:
        import pickle
        tokenizer: tf.keras.preprocessing.text.Tokenizer = pickle.load(f)

    dl = SemevalDataLoader(config, xml_path=input_path)
    processed_sents, _ = dl.get_data()
    processed_sents = tf.keras.preprocessing.sequence.pad_sequences(
        np.asarray(processed_sents), padding='post', value='__PAD__', dtype=object)
    tst_snts = [' '.join(t) for t in processed_sents.tolist()]

    idx_mask = dl.create_mask(tokenizer.word_index, len(processed_sents.tolist()))
    # by default don't pass any thing
    mask_array = np.full(shape=(*processed_sents.shape, model.output[0].shape[2]), fill_value=-np.inf)

    for i in range(len(mask_array)):
        # each sent
        idx_mask_sent = idx_mask[i]
        mask_array_sent = mask_array[i]

        for j in range(len(idx_mask_sent)):
            mask_array_sent[j][idx_mask_sent[j]] = 0.

    preds = model.predict([tst_snts, mask_array])
    preds = np.argmax(preds[0], axis=2)

    raw_sents = dl.get_raw_sentences()
    _, wn2bn = get_bn2wn(resources_path)

    out_f = open(output_path, 'w')
    for i, sent in enumerate(raw_sents):
        for j, word in enumerate(sent):
            if word.instance:
                pred = preds[i][j]
                pred_sense: str = tokenizer.index_word[pred]

                if not pred_sense.startswith('wn:'):  # fallback mfs
                    pred_sense = MFS.predict_word(word.word)

                bn = wn2bn[pred_sense]
                out_f.write(f'{word.iid}\t{bn}\n')

    out_f.close()


def predict_wordnet_domains(input_path: str, output_path: str, resources_path: str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <wordnetDomain>" format (e.g. "d000.s000.t000 sport").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    config = process_config('configs/blstm_config.json')
    # load model
    model: tf.keras.models.Model = tf.keras.models.load_model(join(resources_path, 'model.h5'), custom_objects={
        'ElmoEmbeddingLayer': ElmoEmbeddingLayer,
        'Attention': Attention
    })

    with open(join(resources_path, 'tokenizer.pic'), 'rb') as f:
        import pickle
        tokenizer: tf.keras.preprocessing.text.Tokenizer = pickle.load(f)

    dl = SemevalDataLoader(config, xml_path=input_path)
    processed_sents, _ = dl.get_data()
    processed_sents = tf.keras.preprocessing.sequence.pad_sequences(
        np.asarray(processed_sents), padding='post', value='__PAD__', dtype=object)
    tst_snts = [' '.join(t) for t in processed_sents.tolist()]

    idx_mask = dl.create_mask(tokenizer.word_index, len(processed_sents.tolist()))
    # by default don't pass any thing
    mask_array = np.full(shape=(*processed_sents.shape, model.output[0].shape[2]), fill_value=-np.inf)

    for i in range(len(mask_array)):
        # each sent
        idx_mask_sent = idx_mask[i]
        mask_array_sent = mask_array[i]

        for j in range(len(idx_mask_sent)):
            mask_array_sent[j][idx_mask_sent[j]] = 0.

    preds = model.predict([tst_snts, mask_array])
    preds = np.argmax(preds[0], axis=2)

    raw_sents = dl.get_raw_sentences()
    _, wn2bn = get_bn2wn(resources_path)
    bn2dom, _ = get_bn2dom(resources_path)

    out_f = open(output_path, 'w')
    for i, sent in enumerate(raw_sents):
        for j, word in enumerate(sent):
            if word.instance:
                pred = preds[i][j]
                pred_sense: str = tokenizer.index_word[pred]

                if not pred_sense.startswith('wn:'):  # fallback mfs
                    pred_sense = MFS.predict_word(word.word)

                bn = wn2bn[pred_sense]

                if bn in bn2dom:
                    dom = bn2dom[bn]
                    out_f.write(f'{word.iid}\t{dom}\n')

    out_f.close()


def predict_lexicographer(input_path: str, output_path: str, resources_path: str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <lexicographerId>" format (e.g. "d000.s000.t000 noun.animal").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    config = process_config('configs/blstm_config.json')
    # load model
    model: tf.keras.models.Model = tf.keras.models.load_model(join(resources_path, 'model.h5'), custom_objects={
        'ElmoEmbeddingLayer': ElmoEmbeddingLayer,
        'Attention': Attention
    })

    with open(join(resources_path, 'tokenizer.pic'), 'rb') as f:
        import pickle
        tokenizer: tf.keras.preprocessing.text.Tokenizer = pickle.load(f)

    dl = SemevalDataLoader(config, xml_path=input_path)
    processed_sents, _ = dl.get_data()
    processed_sents = tf.keras.preprocessing.sequence.pad_sequences(
        np.asarray(processed_sents), padding='post', value='__PAD__', dtype=object)
    tst_snts = [' '.join(t) for t in processed_sents.tolist()]

    idx_mask = dl.create_mask(tokenizer.word_index, len(processed_sents.tolist()))
    # by default don't pass any thing
    mask_array = np.full(shape=(*processed_sents.shape, model.output[0].shape[2]), fill_value=-np.inf)

    for i in range(len(mask_array)):
        # each sent
        idx_mask_sent = idx_mask[i]
        mask_array_sent = mask_array[i]

        for j in range(len(idx_mask_sent)):
            mask_array_sent[j][idx_mask_sent[j]] = 0.

    preds = model.predict([tst_snts, mask_array])
    preds = np.argmax(preds[0], axis=2)

    raw_sents = dl.get_raw_sentences()
    _, wn2bn = get_bn2wn(resources_path)
    bn2lex, _ = get_bn2lex(resources_path)

    out_f = open(output_path, 'w')
    for i, sent in enumerate(raw_sents):
        for j, word in enumerate(sent):
            if word.instance:
                pred = preds[i][j]
                pred_sense: str = tokenizer.index_word[pred]

                if not pred_sense.startswith('wn:'):  # fallback mfs
                    pred_sense = MFS.predict_word(word.word)

                bn = wn2bn[pred_sense]
                if bn in bn2lex:
                    lex = bn2lex[bn]
                    out_f.write(f'{word.iid}\t{lex}\n')

    out_f.close()


if __name__ == '__main__':
    predict_babelnet('../WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml', '', '../resources/')
