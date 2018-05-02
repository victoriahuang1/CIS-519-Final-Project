import sys
import os
import pickle
import data_tools as tools
import numpy as np
import language_model as lm
from sklearn import linear_model


lang_to_num = dict(zip(lm.lang_to_fam.keys(), [i for i in range(0, len(lm.lang_to_fam))]))


def get_lm_as_features(lms, examples):
    pickle_path = "pickles/features-{}.p".format(len(examples))
    if os.path.exists(pickle_path):
        return pickle.load(open(pickle_path, "rb"))

    # if pickle not there, then compute
    all_features = np.zeros((len(examples), len(lms)))
    for j in range(0, len(lms)):
        features = lm.predict_labels(examples, lms[j])
        for i in range(0, len(examples)):
            all_features[i][j] = lang_to_num[features[i]]
    pickle.dump(all_features, open("pickles/features-{}.p".format(len(examples)), "wb"))
    return all_features


def predict_by_vote(lms, documents):
    labels = []
    count = 0
    for doc in documents:
        max_lang = -1
        doc_label = '<UNK>'
        lang_count = {}
        for model in lms:
            key = lm.predict_doc_label(doc, model)
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
    print('Loading Test Data...')
    test_data = tools.load_wiki_data(lm.wiki_path + 'test/')

    clf = linear_model.SGDClassifier()

    models = []
    min_n = 2
    max_n = 4

    num_models = max_n - min_n + 1

    print('Training language models...')
    for n in range(2, max_n + 1):
        orders = {}
        for lang in lm.lang_to_fam.keys():
            orders[lang] = n

        models.append(lm.get_language_models(orders, 1, pred_fam, train_data, True))

    train_docs, train_labels = train_data

    sample_size = 1000

    # indices = np.arange(len(train_docs))
    # sample = np.random.choice(indices, size=sample_size, replace=False)

    # train_docs = [train_docs[i] for i in sample]
    # train_labels = [train_labels[i] for i in sample]
    # train_y = np.array(train_labels)


    # test data
    test_docs, gold_labels = test_data

    # indices = np.arange(len(test_docs))
    # sample = np.random.choice(indices, size=sample_size, replace=False)
    #
    # test_docs = [test_docs[i] for i in sample]
    # gold_labels = [gold_labels[i] for i in sample]

    # print('Getting features...')
    # train_x = get_lm_as_features(models, train_docs)
    # test_x = get_lm_as_features(models, test_docs)
    #
    # print('Fitting linear model...')
    # clf.fit(train_x, train_y)
    #
    print('Making Predictions...')
    # y_pred = clf.predict(test_x)

    y_pred = predict_by_vote(models, test_docs)

    if pred_fam:
        for i in range(0, len(gold_labels)):
            gold_labels[i] = lm.lang_to_fam[gold_labels[i]]

    tools.write_pred(y_pred, gold_labels, sys.argv[1])
