from collections import *
import sys
import pickle
import data_tools as tools
import numpy as np
import os

lang_to_fam_path = 'data/languages_to_families.txt'
wiki_path = 'data/wiki/'
wiki_char_file = 'data/data_wiki_chars'
gutenberg_path = 'data/gutenberg/'
gutenberg_char_file = 'data/data_gutenberg_chars/'

# maps a language to its language family
lang_to_fam = tools.read_lang_to_fam(lang_to_fam_path)

# all possible chars in train and test datasets
vocab = set([chr(i) for i in range(256)])

# vocab = tools.load_all_chars(wiki_char_file)


class LanguageModel(object):
    def __init__(self, language, order=4, add_k=1.0, learn_fam=False, pretrained=False, corpus = 'wiki'):
        super(LanguageModel, self).__init__()

        self.language = language
        self.family = lang_to_fam[language]
        self.order = order
        self.add_k = add_k
        # do we want the model to predict languages or families?
        self.learn_fam = learn_fam
        self.ngrams = {}
        self.pretrained = pretrained
        self.corpus = 'wiki'
    # trains model on training data (should take in list of documents)
    def train(self, train_data):
        if self.pretrained:
            pickle_path = "pickles/{}-{}-{}.p".format(self.language, self.order, self.corpus)
            if os.path.exists(pickle_path):
                self.ngrams = pickle.load(open(pickle_path, "rb"))
                return

        lm = defaultdict(Counter)
        pad = "~" * self.order
        lm['<UNK>'] = {}

        words, labels = train_data

        count = 0
        for j in range(0, len(words)):
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

            count += 1
            #TODO: change?
            if count == 10:
                break

        # TODO: handle families
        # vocab = tools.load_all_chars(wiki_char_file + "_" + self.language)

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
        print("Finished {}".format(self.language))
        pickle.dump(self.ngrams, open("pickles/{}-{}-{}.p".format(self.language, self.order, self.corpus), "wb"))

    # gives perplexity score
    def predict(self, document):
        probs_list = []
        i = 0
        while i < len(document):
            full_n_gram = document[i:(self.order + 1 + i)]
            history = full_n_gram[:len(full_n_gram) - 1]
            char = full_n_gram[-1]

            if history not in self.ngrams:
                history = '<UNK>'
            probs_tuple = self.ngrams[history]
            chars, probs = zip(*probs_tuple)
            if char in chars:
                probIndex = chars.index(char)
                prob = probs[probIndex]

                probs_list.append(prob)
            i += 1
        if i == 0:
            return float("inf")

        # use perplexity since it normalizes with the number of words (some characters might be OOV in some languages)
        perplexity = np.sum(np.ma.log(probs_list)) * (-1.0 / i)

        return perplexity


# predicts the label of a single document
def predict_doc_label(doc, models):
    min_perplexity = float("inf")
    idx = 0
    doc_text = ' '.join(doc[:100])
    for i in range(0, len(models)):
        perplexity = models[i].predict(doc_text)
        if perplexity < min_perplexity:
            min_perplexity = perplexity
            idx = i
    if models[idx].learn_fam:
        return models[idx].family
    else:
        return models[idx].language


# returns a list of predictions on the languages for each document
def predict_labels(documents, models):
    labels = []
    count = 0
    for doc in documents:
        labels.append(predict_doc_label(doc, models))
        count += 1

        if count % 500 == 0:
            print("Labels predicted: {}".format(count))
    return labels


def predict_by_word(documents, models):
    labels = []
    count = 0
    for doc in documents:
        max_lang = -1
        doc_label = '<UNK>'
        lang_count = {}
        for word in doc[:100]:
            min_perplexity = float("inf")
            idx = 0
            for i in range(0, len(models)):
                perplexity = models[i].predict(word)
                if perplexity < min_perplexity:
                    min_perplexity = perplexity
                    idx = i

            if models[idx].learn_fam:
                key = models[idx].family
            else:
                key = models[idx].language

            if key not in lang_count:
                lang_count[key] = 0
            lang_count[key] += 1
            if lang_count[key] > max_lang:
                max_lang = lang_count[key]
                doc_label = key

        labels.append(doc_label)
        count += 1

        if count % 500 == 0:
            print("Labels predicted: {}".format(count))
    return labels


def get_language_models(orders, add_k, learn_fam, train_data, pretrained, corpus = 'wiki'):
    models = []

    # currently assuming that P(Country) is the same for all countries - TODO: CHANGE?
    for language in lang_to_fam.keys():
        print("Training {} model...".format(language))
        model = LanguageModel(language, order=orders[language], add_k=add_k, learn_fam=learn_fam, pretrained=pretrained, corpus=corpus)
        model.train(train_data)
        models.append(model)
    return models


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Argument for [language model order] and [output file] needed')

    pred_fam = False
    pred_lang = False

    if len(sys.argv) > 3:
        pred_fam = True
    if len(sys.argv) > 4:
        pred_lang = True

    print('Loading Train Data...')
    train_data = tools.load_wiki_data(wiki_path + 'train/')
    print('Loading Test Data...')
    test_data = tools.load_wiki_data(wiki_path + 'test/')

    n = int(sys.argv[1])
    orders = {}
    for lang in lang_to_fam.keys():
        orders[lang] = n

    print('Training language models...')
    models = get_language_models(orders, 1, pred_fam, train_data, True)

    # test data
    docs, gold_labels = test_data

    print('Making Predictions...')
    y_pred = predict_labels(docs, models)
    # y_pred = predict_by_word(docs, models)

    if pred_fam:
        for i in range(0, len(gold_labels)):
            gold_labels[i] = lang_to_fam[gold_labels[i]]

    tools.write_pred(y_pred, gold_labels, sys.argv[2])
