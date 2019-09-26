import numpy as np
from nltk.corpus import wordnet as wn


class MFS:
    @staticmethod
    def predict_word(word):
        synsets = wn.synsets(word)

        if synsets is None or len(synsets) == 0:
            return word

        synset = synsets[0]  # fetch MFS
        return 'wn:' + str(synset.offset()).zfill(8) + synset.pos()

    @staticmethod
    def predict_sentence(sentence):
        return [
            MFS.predict_word(w) for w in sentence
        ]

    @staticmethod
    def predict_batch(batch):
        return [
            MFS.predict_sentence(s) for s in batch
        ]
