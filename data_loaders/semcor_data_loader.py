from ast import literal_eval

from data_loaders import BaseRaganatoDataLoader, TrainWord
from utils.conversion import *


class SemcorDataLoader(BaseRaganatoDataLoader):
    def __init__(self, config):
        super().__init__(config)

        self.xml_path = 'WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml'
        self.gold_path = 'WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt'

    def __process_dataset(self):
        batch_size = int(self.config.trainer.batch_size)

        if len(self.gold_path) != 0:
            lbls = self._get_labels()
        _, wn2bn = get_bn2wn(self.config.data.resources_path)
        bn2lex, _ = get_bn2lex(self.config.data.resources_path)

        processed_sents, sense_sents, pos_sents, lex_sents = [], [], [], []

        for sent in self._get_sentences():
            # build normal sentence
            processed_sent = []
            # build annotated sentences
            sense_annotated, pos_annotated, lex_annotated = [], [], []

            for w in sent:
                processed_sent.append(w.word)
                if len(self.gold_path) != 0:
                    pos_annotated.append(w.pos)

                    if not w.instance:
                        sense_annotated.append(w.word)
                        lex_annotated.append('other')
                    else:
                        senses = lbls[w.iid]
                        # TODO: what if more than one sense
                        sense_annotated.append(senses[0])
                        lex_annotated.append(bn2lex[wn2bn[senses[0]]])

            processed_sents.append(processed_sent)
            if len(self.gold_path) != 0:
                sense_sents.append(sense_annotated)
                pos_sents.append(pos_annotated)
                lex_sents.append(lex_annotated)

        while len(processed_sents) % batch_size != 0:
            processed_sents.append(processed_sents[0])
            if len(self.gold_path) != 0:
                sense_sents.append(sense_sents[0])
                pos_sents.append(pos_sents[0])
                lex_sents.append(lex_sents[0])

        return processed_sents, [sense_sents, pos_sents, lex_sents]

    def get_senses(self):
        lbls = self._get_labels()
        senses = set()

        for v in lbls.values():
            senses = senses.union(set(v))

        return list(senses)

    def get_data(self):
        return self.__process_dataset()

    def read_data(self):
        processed_sents, sense_sents, pos_sents, lex_sents = [], [], [], []

        with open(join(self.config.data.resources_path, 'train_sentences.txt')) as f:
            for line in f:
                processed_sents.append(literal_eval(line))

        with open(join(self.config.data.resources_path, 'sense_sentences.txt')) as f:
            for line in f:
                sense_sents.append(literal_eval(line))

        with open(join(self.config.data.resources_path, 'pos_sentences.txt')) as f:
            for line in f:
                pos_sents.append(literal_eval(line))

        with open(join(self.config.data.resources_path, 'lex_sentences.txt')) as f:
            for line in f:
                lex_sents.append(literal_eval(line))

        return processed_sents, [sense_sents, pos_sents, lex_sents]

    def write_data(self, data):
        processed_sents, (sense_sents, pos_sents, lex_sents) = data

        with open(join(self.config.data.resources_path, 'train_sentences.txt'), 'w') as f:
            for sent in processed_sents:
                f.write('{}\n'.format(sent))

        with open(join(self.config.data.resources_path, 'sense_sentences.txt'), 'w') as f:
            for sent in sense_sents:
                f.write('{}\n'.format(sent))

        with open(join(self.config.data.resources_path, 'pos_sentences.txt'), 'w') as f:
            for sent in pos_sents:
                f.write('{}\n'.format(sent))

        with open(join(self.config.data.resources_path, 'lex_sentences.txt'), 'w') as f:
            for sent in lex_sents:
                f.write('{}\n'.format(sent))

    def read_senses(self):
        senses = set()

        with open(join(self.config.data.resources_path, 'senses.txt')) as f:
            for line in f:
                senses.add(line.strip())

        return list(senses)

    @staticmethod
    def get_pos():
        return ['.', 'adj', 'adp', 'adv', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x']

    @staticmethod
    def get_lexes():
        return [
            'adj.all', 'adj.pert', 'adj.ppl', 'adv.all',

            'noun.Tops', 'noun.act', 'noun.animal', 'noun.artifact', 'noun.attribute', 'noun.body', 'noun.cognition',
            'noun.communication', 'noun.event', 'noun.feeling', 'noun.food', 'noun.group', 'noun.location', 'noun.motive',
            'noun.object', 'noun.person', 'noun.phenomenon', 'noun.plant', 'noun.possession', 'noun.process', 'noun.quantity',
            'noun.relation', 'noun.shape', 'noun.state', 'noun.substance', 'noun.time',

            'verb.body', 'verb.change', 'verb.cognition', 'verb.communication', 'verb.competition', 'verb.consumption', 'verb.contact',
            'verb.creation', 'verb.emotion', 'verb.motion', 'verb.perception', 'verb.possession', 'verb.social', 'verb.stative', 'verb.weather',

            'other'
        ]

    def create_mask(self, d, n):
        sentences = self._get_sentences()

        mask_array = []

        for sent in sentences:
            mask_sent = []

            word: TrainWord
            for word in sent:
                mask_word = []
                # append senses only
                if word.instance:
                    for sense in SemcorDataLoader.get_candidate_synsets(word.word):
                        if sense in d:
                            mask_word.append(d[sense])
                else:  # append wf only
                    if word.word in d:
                        mask_word.append(d[word.word])
                    else:
                        mask_word.append(d['<UNK>'])

                mask_sent.append(mask_word)

            mask_array.append(mask_sent)

        while len(mask_array) < n:
            mask_array.append(mask_array[0])

        return mask_array

    @staticmethod
    def get_candidate_synsets(word):
        from nltk.corpus import wordnet as wn

        candidates = []

        for synset in wn.synsets(lemma=word):
            sense = 'wn:' + str(synset.offset()).zfill(8) + synset.pos()
            candidates.append(sense)

        return candidates


if __name__ == '__main__':
    pass
