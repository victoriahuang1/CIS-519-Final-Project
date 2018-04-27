from random import random


# maps a language to its language family
lang_to_fam =
    {'English':'Indo-European',
     'Hebrew':'Afro-Asiatic',
     'Arabic':'Afro-Asiatic',
     'Spanish':'Indo-European',
     'French':'Indo-European',
     'German':'Indo-European',
     'Afrikaans':'Indo-European',
     'Swahili':'Niger–Congo',
     'Zulu':'Niger–Congo',
     'Chinese':'Sino-Tibetan'}


# Just a simple baseline that should do roughly as well as just randomly guessing the label
class BaselineModel(object):
    def __init__(self):
        super(BaselineModel, self).__init__()

        # maps a language to the proportion of documents written in this language
        # will be populated in train
        self.lang_prob = {}

        # maps a language family to the proportion of documents written in a language of that
        # family
        self.fam_prob = dict(zip(lang_to_fam.keys(), [0 for i in range(len(lang_to_fam)))]))

    # trains model on training data
    def train(train_data):
        # maps the labels (languages) to the number of documents written in that language
        lang_count = {}
        # total number of documents
        total = 0.0
        # TODO: fill in language_count dict (what is the format of the data?)

        for key, val in lang_count:
            self.lang_prob[key] = val / total
            self.fam_prob[lang_to_fam[key]] += self.lang_prob[key]

    # predict language using computed distribution
    def predict_language(document):
        x = random()

        for lang, prob in self.lang_prob:
            x = x - prob
            if x <= 0:
                return lang

    # predicts the language family that the language a text is written in belongs to
    # simple baseline - just uses the proportion of documents written in that language
    def predict_family(document, predict_lang=False):
        if predict_lang:
            return lang_to_fam[predict_language(document)]
        x = random()

        for fam, prob in self.fam_prob:
            x = x - prob
            if x <= 0:
                return fam

def write_pred(train_data, test_data, out_path, pred_fam=False):
    model = Baseline()
    model.train(train_data)

    with open(out_path, 'w') as out:
        for document in test_data:
            if pred_fam:
                out.write(predict_family(document) + '\n')
            else:
                out.write(predict_language(document) + '\n')

