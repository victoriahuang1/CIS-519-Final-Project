from collections import *
import sys
import pprint
import data_tools as tools
import numpy as np

lang_to_fam_path = 'data/languages_to_families.txt'
wiki_path = 'data/wiki/'
wiki_char_file = 'data/wiki_all_chars'

# maps a language to its language family
lang_to_fam = tools.read_lang_to_fam(lang_to_fam_path)

# all possible chars in train and test datasets
vocab = tools.load_all_chars(wiki_char_file)


def print_probs(lm, history):
    probs = sorted(lm[history], key=lambda x: (-x[1], x[0]))
    pp = pprint.PrettyPrinter()
    pp.pprint(probs)


class LanguageModel(object):
    def __init__(self, language, order=4, add_k=1, learn_fam=False):
        super(LanguageModel, self).__init__()

        self.language = language
        self.family = lang_to_fam[language]
        self.order = order
        self.add_k = add_k
        # do we want the model to predict languages or families?
        self.learn_fam = learn_fam
        self.ngrams = {}

    # trains model on training data (should take in list of documents)
    def train(self, train_data):
        lm = defaultdict(Counter)
        pad = "~" * self.order
        lm['<UNK>'] = {}

        words, labels = train_data

        for j in range(0, len(train_data)):
            # skip if data not for this language/family
            if (self.learn_fam and self.family != lang_to_fam[labels[j]]) or \
                    ((not self.learn_fam) and self.language != labels[j]):
                continue

            # for now, we're not going to predict the words one by one
            doc_text = ' '.join(words[j])

            data = pad + doc_text

            for i in range(len(data) - self.order):
                history, char = data[i:i + self.order], data[i + self.order]
                lm[history][char] += 1

        # given the history, what is the prob of the next letter?
        for key in lm.keys():
            for v in vocab:
                if v not in lm[key]:
                    lm[key][v] = 0

        def normalize(counter):
            s = float(sum(counter.values()))
            return [(c, (cnt + self.add_k) / (s + self.add_k * len(vocab))) for c, cnt in counter.items()]

        for hist, chars in lm.items():
            self.ngrams[hist] = normalize(chars)

    # predict language or family depending on how model was initialized and trained
    def predict(self, document):
        if self.learn_fam:
            return self.language[0]

        probs_list = []
        N = 0
        i = 0
        while i < len(document):
            full_n_gram = document[i:(self.order + 1 + i)]
            history = full_n_gram[:len(full_n_gram) - 1]
            char = full_n_gram[-1]

            if history not in self.ngrams:
                history = '<UNK>'
            probs_tuple = self.ngrams[history]
            chars, probs = zip(*probs_tuple)
            probIndex = chars.index(char)
            prob = probs[probIndex]

            probs_list.append(prob)
            i += 1
            N += 1
        perplexity = np.sum(np.ma.log(probs_list)) * (-1 / N)

        return perplexity


# returns a list of preditions on the languages for each document
def predict_labels(documents, models):
    print('Making Predictions...')
    labels = []
    for doc in documents:
        min_perplexity = float("inf")
        idx = 0
        for i in range(0, len(models)):
            perplexity = models[i].predict(' '.join(doc))
            if perplexity < min_perplexity:
                min_perplexity = perplexity
                idx = i
        if models[idx].learn_fam:
            labels.append(models[idx].family)
        else:
            labels.append(models[idx].language)
    return labels


# predicts language that a text is written in
def text_classification(train_data, test_data, orders, add_k=1, learn_fam=False):
    models = []

    # currently assuming that P(Country) is the same for all countries - TODO: CHANGE?
    for language in lang_to_fam.keys():
        model = LanguageModel(language, order=orders[language], add_k=add_k, learn_fam=learn_fam)
        model.train(train_data)
        models.append(model)

    # test data
    docs, gold_labels = test_data
    y_pred = predict_labels(docs, models)

    return y_pred, gold_labels


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Argument for output file needed')

    pred_fam = False
    pred_lang = False

    if len(sys.argv) > 2:
        pred_fam = True
    if len(sys.argv) > 3:
        pred_lang = True

    print('Loading Train Data...')
    train_data = tools.load_wiki_data(wiki_path + 'train/')
    print('Loading Test Data...')
    test_data = tools.load_wiki_data(wiki_path + 'test/')

    n = 4
    orders = {}
    for lang in lang_to_fam.keys():
        orders[lang] = n

    print('Training language models...')
    y_pred, gold_labels = text_classification(train_data, test_data, orders, add_k=1, learn_fam=pred_fam)

    if pred_fam:
        for i in range(0, len(gold_labels)):
            gold_labels[i] = lang_to_fam[gold_labels[i]]

    tools.write_pred(y_pred, gold_labels, sys.argv[1])
