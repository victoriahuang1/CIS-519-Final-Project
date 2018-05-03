import sys
import os
import pickle
import data_tools as tools
import numpy as np
import language_model as lm
from sklearn import linear_model
from sklearn.svm import SVC


lang_to_num = dict(zip(lm.lang_to_fam.keys(), [i for i in range(0, len(lm.lang_to_fam))]))


def get_lm_as_features(lms, documents, is_train=True):
    # Do not pickle: train labels may not match up
    # pickle_path = "pickles/features.p"
    # if os.path.exists(pickle_path) and is_train:
        # return pickle.load(open(pickle_path, "rb"))

    # if pickle not there, then compute
    count = 0
    all_features = np.zeros((len(documents), len(lms) + 2))
    for i in range(0, len(documents)):
        doc_text = ' '.join(documents[i])
        for j in range(0, len(lms)):
            score = lms[j].predict(doc_text)
            all_features[i][j] = score
        all_features[i][len(lms)] = 1 if all(ord(char) < 128 for char in doc_text) else 0
        all_features[i][len(lms) + 1] = float(len(doc_text)) / len(documents[i])    # average word length
        count += 1
        if count % 50 == 0:
            print("Features extracted: {}".format(count))

    # if is_train:
    #     pickle.dump(all_features, open("pickles/features.p", "wb"))
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
    vote = False

    if len(sys.argv) > 2:
        vote = int(sys.argv[2]) == 1
    if len(sys.argv) > 3:
        pred_fam = True

    print('Loading Wiki Train Data...')
    train_data = tools.load_wiki_data(lm.wiki_path + 'train/')
    train_docs, train_labels = train_data
    print('Loading Wiki Test Data...')
    test_data = tools.load_wiki_data(lm.wiki_path + 'test/')
    test_docs, gold_labels = test_data

    print('Loading Gutenberg Train Data...')
    train_data_2 = tools.load_gutenberg_data(lm.gutenberg_path + 'train/')
    train_docs_2, train_labels_2 = train_data_2
    print('Loading Gutenberg Test Data...')
    test_data_2 = tools.load_gutenberg_data(lm.gutenberg_path + 'test/')
    test_docs_2, gold_labels_2 = test_data_2
    min_order = 2
    max_order = 4

    print('Training language models...')

    if vote:
        ensemble = Ensemble(min_order, max_order, learn_fam=pred_fam)
        ensemble.train(train_data)
        
        ensemble_2 = Ensemble(min_order, max_order, learn_fam=pred_fam)
        ensemble_2.train(train_data_2)

        print('Making Predictions...')
        y_pred = predict_by_vote(ensemble, test_docs)
        y_pred_2 = predict_by_vote(ensemble_2, test_docs_2)
    else:
        # clf = linear_model.SGDClassifier()
        clf = SVC()
        clf_2 = SVC()
        
        sample_size = 500

        indices = np.arange(len(train_docs))
        sample = np.random.choice(indices, size=sample_size, replace=False)

        train_docs = [train_docs[i] for i in sample]
        train_labels = [train_labels[i] for i in sample]

        indices_2 = np.arange(len(train_docs_2))
        sample_2 = np.random.choice(indices_2, size=max(sample_size, indices_2), replace=False)

        train_docs = [train_docs[i] for i in sample]
        train_labels = [train_labels[i] for i in sample]
        
        train_docs_2 = [train_docs_2[i] for i in sample_2]
        train_labels_2 = [train_labels_2[i] for i in sample_2]
        
        if pred_fam:
            for i in range(0, len(train_docs)):
                train_labels[i] = lm.lang_to_fam[train_labels[i]]
                
            for i in range(0, len(train_docs_2)):
                train_labels_2[i] = lm.lang_to_fam[train_labels_2[i]]

        indices = np.arange(len(test_docs))
        sample = np.random.choice(indices, size=sample_size, replace=False)

        indices_2 = np.arange(len(test_docs_2))
        sample_2 = np.random.choice(indices_2, size=max(sample_size, indices_2), replace=False)
        
        test_docs = [test_docs[i] for i in sample]
        gold_labels = [gold_labels[i] for i in sample]
        
        test_docs_2 = [test_docs_2[i] for i in sample_2]
        gold_labels_2 = [gold_labels[i] for i in sample_2]

        n = 3
        orders = {}
        for lang in lm.lang_to_fam.keys():
            orders[lang] = n

        lms = lm.get_language_models(orders, 1, pred_fam, train_data, True)
        lms_2 = lm.get_language_models(orders, 1, pred_fam, train_data_2, True)
        
        print('Getting features...')
        train_x = get_lm_as_features(lms, train_docs)
        test_x = get_lm_as_features(lms, test_docs, is_train=False)
        
        train_x_2 = get_lm_as_features(lms_2, train_docs_2)
        test_x_2 = get_lm_as_features(lms_2, test_docs_2)

        print('Fitting linear model...')
        train_y = np.array(train_labels)
        clf.fit(train_x, train_y)

        train_y_2 = np.array(train_labels_2)
        clf_2.fit(train_x_2, train_y_2)
        
        print('Making Predictions...')
        y_pred = clf.predict(test_x)
        
        y_pred = clf_2.predict(test_x_2)

    if pred_fam:
        for i in range(0, len(gold_labels)):
            gold_labels[i] = lm.lang_to_fam[gold_labels[i]]
        
        for i in range(0, len(gold_labels_2)):
            gold_labels_2[i] = lm.lang_to_fam[gold_labels_2[i]]

    tools.write_pred(y_pred, gold_labels, sys.argv[1])
    tools.write_pred(y_pred_2, gold_labels_2, (sys.argv[1] + '_1'))
