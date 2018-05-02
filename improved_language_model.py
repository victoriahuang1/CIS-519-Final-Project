import sys
import os
import pickle
import data_tools as tools
import numpy as np
import language_model as lm
from sklearn import linear_model


lang_to_num = dict(zip(lm.lang_to_fam.keys(), [i for i in range(0, len(lm.lang_to_fam))]))


def get_lm_as_features(lms, documents, is_train=True):
    pickle_path = "pickles/features-{}.p".format('tr' if is_train else 'ts')
    if os.path.exists(pickle_path):
        return pickle.load(open(pickle_path, "rb"))

    # if pickle not there, then compute
    count = 0
    all_features = np.zeros((len(documents), len(lms)))
    for i in range(0, len(documents)):
        for j in range(0, len(lms)):
            score = lms[j].predict(' '.join(documents[i]))
            all_features[i][j] = score
        count += 1
        if count % 10 == 0:
            print("Features extracted: {}".format(count))

    pickle.dump(all_features, open("pickles/features-{}.p".format('tr' if is_train else 'ts'), "wb"))
    return all_features


class Ensemble(object):
    def __init__(self, min_order, max_order, add_k=1.0, learn_fam=False):
        self.min_order = min_order
        self.max_order = max_order
        self.add_k = add_k
        self.learn_fam = learn_fam
        self.models = []

    def train(self, train_data):
        for n in range(self.min_order, self.max_order + 1):
            orders = {}
            for lang in lm.lang_to_fam.keys():
                orders[lang] = n

            self.models.append(lm.get_language_models(orders, self.add_k, self.learn_fam, train_data, True))

    # all models vote on the label, and winner takes all
    def predict(self, doc):
        max_lang = -1
        doc_label = '<UNK>'
        lang_count = {}
        for model in self.models:
            key = lm.predict_doc_label(doc, model)
            if key not in lang_count:
                lang_count[key] = 0
            lang_count[key] += 1
            if lang_count[key] > max_lang:
                max_lang = lang_count[key]
                doc_label = key
        return doc_label


def predict_by_vote(ensemble, documents):
    labels = []
    count = 0
    for doc in documents:
        labels.append(ensemble.predict(doc))
        count += 1

        if count % 500 == 0:
            print("Labels predicted: {}".format(count))
    return labels


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
    train_data = tools.load_wiki_data(lm.wiki_path + 'train/')
    train_docs, train_labels = train_data
    print('Loading Test Data...')
    test_data = tools.load_wiki_data(lm.wiki_path + 'test/')
    test_docs, gold_labels = test_data

    min_order = 2
    max_order = 4

    vote = False

    print('Training language models...')

    if vote:
        ensemble = Ensemble(min_order, max_order, learn_fam=pred_fam)
        ensemble.train(train_data)

        print('Making Predictions...')
        y_pred = predict_by_vote(ensemble, test_docs)
    else:
        clf = linear_model.SGDClassifier()

        sample_size = 100

        indices = np.arange(len(train_docs))
        sample = np.random.choice(indices, size=sample_size, replace=False)

        train_docs = [train_docs[i] for i in sample]
        train_labels = [train_labels[i] for i in sample]

        indices = np.arange(len(test_docs))
        sample = np.random.choice(indices, size=sample_size, replace=False)

        test_docs = [test_docs[i] for i in sample]
        gold_labels = [gold_labels[i] for i in sample]

        n = 3
        orders = {}
        for lang in lm.lang_to_fam.keys():
            orders[lang] = n

        lms = lm.get_language_models(orders, 1, pred_fam, train_data, True)

        print('Getting features...')
        train_x = get_lm_as_features(lms, train_docs)
        test_x = get_lm_as_features(lms, test_docs, is_train=False)

        print('Fitting linear model...')
        train_y = np.array(train_labels)
        clf.fit(train_x, train_y)

        print('Making Predictions...')
        y_pred = clf.predict(test_x)

    if pred_fam:
        for i in range(0, len(gold_labels)):
            gold_labels[i] = lm.lang_to_fam[gold_labels[i]]

    tools.write_pred(y_pred, gold_labels, sys.argv[1])
