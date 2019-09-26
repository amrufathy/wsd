import re
import string
from collections import namedtuple
from functools import lru_cache

from lxml.etree import iterparse
from nltk.corpus import wordnet as wn

from base import BaseDataLoader

TrainWord = namedtuple('TrainWord', 'iid word pos instance')


class BaseRaganatoDataLoader(BaseDataLoader):
    def __init__(self, config):
        super().__init__(config)

        self.xml_path = None
        self.gold_path = None

    def _get_sentences(self):
        context = iterparse(self.xml_path, events=('start', 'end'))

        for (event, element) in context:
            sent = []

            if element.tag == 'sentence' and event == 'end':
                for child in element:
                    _id = child.get('id') or None
                    pos = child.get('pos').lower()
                    word = child.get('lemma').lower()

                    if word == '':
                        continue

                    sent.append(TrainWord(iid=_id, word=word, pos=pos, instance=(child.tag == 'instance')))

                yield sent

                element.clear()

    @lru_cache()
    def _get_labels(self):
        lbls = dict()
        with open(self.gold_path) as f:
            for idx, line in enumerate(f):
                line = line.strip().split()
                _id, sense_keys = line[0], line[1:]

                senses = set()
                for s_key in sense_keys:
                    synset = wn.lemma_from_key(s_key).synset()
                    synset_id = 'wn:' + str(synset.offset()).zfill(8) + synset.pos()

                    senses.add(synset_id)

                lbls[_id] = list(senses)

        return lbls

    @staticmethod
    def __clean_word(s):
        """
        Removes punctuation and multiple consecutive
          spaces from text
        """
        # remove punctuation characters
        s = s.translate(
            str.maketrans('', '', string.punctuation.replace('_', '')))
        # remove multiple consecutive spaces
        s = re.sub(' +', ' ', s)

        return s.lower()

    def get_raw_sentences(self):
        return list(self._get_sentences())

    def get_data(self):
        raise NotImplementedError


if __name__ == '__main__':
    pass
