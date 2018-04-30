from random import random
import sys
import data_tools as tools

lang_to_fam_path = 'data/languages_to_families.txt'
wiki_path = 'data/wiki/'


# maps a language to its language family
lang_to_fam = tools.read_lang_to_fam(lang_to_fam_path)


# Just a simple baseline that should do roughly as well as just randomly guessing the label
class BaselineModel(object):
    def __init__(self):
        super(BaselineModel, self).__init__()

        # maps a language to the proportion of documents written in this language
        # will be populated in train
        self.lang_prob = {}

        families = list(set(lang_to_fam.values()))

        # maps a language family to the proportion of documents written in a language of that
        # family
        self.fam_prob = dict(zip(families, [0 for i in range(len(families))]))

    # trains model on training data
    def train(self, train_data):
        _, labels = train_data

        # maps the labels (languages) to the number of documents written in that language
        lang_count = {}
        # total number of documents
        total = float(len(labels))

        for lang in labels:
            if lang not in lang_count:
                lang_count[lang] = 0
            lang_count[lang] += 1

        for key, val in lang_count.items():
            self.lang_prob[key] = val / total
            self.fam_prob[lang_to_fam[key]] += self.lang_prob[key]

    # predict language using computed distribution
    def predict_language(self, document):
        x = random()

        for lang, prob in self.lang_prob.items():
            x = x - prob
            if x <= 0:
                return lang

    # predicts the language family that the language a text is written in belongs to
    # simple baseline - just uses the proportion of documents written in that language
    def predict_family(self, document, predict_lang=False):
        if predict_lang:
            return lang_to_fam[self.predict_language(document)]
        x = random()

        for fam, prob in self.fam_prob.items():
            x = x - prob
            if x <= 0:
                return fam


def write_pred(train_data, test_data, out_path, pred_fam=False, pred_lang=False):
    model = BaselineModel()
    model.train(train_data)

    docs, labels = test_data

    i = 0
    with open(out_path, 'w') as out:
        for doc in docs:
            pred = model.predict_language(doc)
            if pred_fam:
                pred = model.predict_family(doc, predict_lang=pred_lang)
                labels[i] = lang_to_fam[labels[i]]
            out.write("{}\t{}\n".format(pred, labels[i]))
            i += 1


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Argument for output file needed')

    pred_fam = False
    pred_lang = False

    if len(sys.argv) > 2:
        pred_fam = True
    if len(sys.argv) > 3:
        pred_lang = True

    train_data = tools.load_wiki_data(wiki_path + 'train/')
    test_data = tools.load_wiki_data(wiki_path + 'test/')

    write_pred(train_data, test_data, sys.argv[1], pred_fam=pred_fam, pred_lang=pred_lang)
