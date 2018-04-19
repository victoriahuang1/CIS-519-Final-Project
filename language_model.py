from collections import *
from random import random
import pprint
import operator
import numpy as np
from itertools import product


def print_probs(lm, history):
    probs = sorted(lm[history], key=lambda x: (-x[1], x[0]))
    pp = pprint.PrettyPrinter()
    pp.pprint(probs)


def train_char_lm(fname, order=4, add_k=1):
    ''' Trains a language model.

    This code was borrowed from
    http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139

    Inputs:
      fname: Path to a text corpus.
      order: The length of the n-grams.
      add_k: k value for add-k smoothing. NOT YET IMPLMENTED

    Returns:
      A dictionary mapping from n-grams of length n to a list of tuples.
      Each tuple consists of a possible net character and its probability.
    '''

    data = open(fname, errors='ignore').read()
    lm = defaultdict(Counter)
    pad = "~" * order
    data = pad + data
    lm['<UNK>'] = dict()
    for i in range(len(data) - order):
        history, char = data[i:i + order], data[i + order]
        lm[history][char] += 1

    # piazza 329 says we can make this assumption
    vocab = set([chr(i) for i in range(128)])
    # vocab = set(list(data))
    for key in lm.keys():
        for v in vocab:
            if v not in lm[key]:
                lm[key][v] = 0

    def normalize(counter):
        s = float(sum(counter.values()))
        return [(c, (cnt + add_k) / (s + add_k * len(vocab))) for c, cnt in counter.items()]

    outlm = {hist: normalize(chars) for hist, chars in lm.items()}
    return outlm


def generate_letter(lm, history, order):
    ''' Randomly chooses the next letter using the language model.

    Inputs:
      lm: The output from calling train_char_lm.
      history: A sequence of text at least 'order' long.
      order: The length of the n-grams in the language model.

    Returns:
      A letter
    '''

    history = history[-order:]
    if history not in lm:
        history = '<UNK>'
    dist = lm[history]
    x = random()
    for c, v in dist:
        x = x - v
        if x <= 0: return c


def generate_text(lm, order, nletters=500):
    '''Generates a bunch of random text based on the language model.

    Inputs:
    lm: The output from calling train_char_lm.
    history: A sequence of previous text.
    order: The length of the n-grams in the language model.

    Returns:
      A letter
    '''
    history = "~" * order
    out = []
    for i in range(nletters):
        c = generate_letter(lm, history, order)
        history = history[-order:] + c
        out.append(c)
    return "".join(out)


def perplexity(test_filename, lm, order=4):
    '''Computes the perplexity of a text file given the language model.

    Inputs:
      test_filename: path to text file
      lm: The output from calling train_char_lm.
      order: The length of the n-grams in the language model.
    '''
    test = open(test_filename, errors='ignore').read()
    pad = "~" * order
    test = pad + test
    probs_list = []
    N = 0
    i = 0
    while i < len(test):
        full_n_gram = test[i:(order + 1 + i)]
        history = full_n_gram[:len(full_n_gram) - 1]
        if history not in lm:
            history = '<UNK>'
        # if history in lm:
        #     probs_tuple = lm[history]
        # else:
        #     i += 1
        #     N += 1
        #     continue
        probs_tuple = lm[history]
        chars, probs = zip(*probs_tuple)
        probIndex = chars.index(full_n_gram[len(full_n_gram) - 1])
        probs_list.append(probs[probIndex])

        i += 1
        N += 1
    probs_list = np.array(probs_list)
    return np.sum(np.ma.log(probs_list)) * (-1 / N)


def perplexity_backoff(test_filename, lms, lambdas, order=4):
    '''Computes the perplexity of a text file given the language models. (Used to tune interpolation lambdas)

    Inputs:
      test_filename: path to text file
      lms: A list of language models, outputted by calling train_char_lm.
      order: The length of the n-grams in the language model.
    '''

    test = open(test_filename, errors='ignore').read()
    pad = "~" * order
    test = pad + test
    probs_list = []
    N = 0
    i = 0
    while i < len(test):
        full_n_gram = test[i:(order + 1 + i)]
        history = full_n_gram[:len(full_n_gram) - 1]
        prob = calculate_prob_with_backoff(full_n_gram[-1], history, lms, lambdas)
        probs_list.append(prob)
        i += 1
        N += 1
    probs_list = np.array(probs_list)
    return np.sum(np.ma.log(probs_list)) * (-1 / N)


def calculate_prob_with_backoff(char, history, lms, lambdas):
    '''Uses interpolation to compute the probability of char given a series of
       language models trained with different length n-grams.

     Inputs:
       char: Character to compute the probability of.
       history: A sequence of previous text.
       lms: A list of language models, outputted by calling train_char_lm.
       lambdas: A list of weights for each lambda model. These should sum to 1.

    Returns:
      Probability of char appearing next in the sequence.
    '''
    # want to find probability of char given history
    # lms of different orders, each with a lambda

    prob = 0
    for i in range(0, len(lms)):
        curr_prob = 0
        hist = history[len(history) - i:]
        # if hist not in lms[i]:
        #     hist = '<UNK>'
        if hist in lms[i]:
            probs_tuple = lms[i][hist]
            chars, probs = zip(*probs_tuple)
            probIndex = chars.index(char)
            curr_prob = probs[probIndex]
        prob += lambdas[i] * curr_prob

    return prob


def set_lambdas(lms, dev_filename):
    '''Returns a list of lambda values that weight the contribution of each n-gram model

    This can either be done heuristically or by using a development set.

    Inputs:
      lms: A list of language models, outputted by calling train_char_lm.
      dev_filename: Path to a development text file to optionally use for tuning the lmabdas.

    Returns:
      Probability of char appearing next in the sequence.
    '''

    incr = 0.1

    # all possible values of all lambdas
    vals = [[j * incr for j in range(0, int(1.0 / incr))] for i in range(0, len(lms))]


    # initial values for lambdas - may change after tuning
    lambdas = [0.0 for i in range(0, len(lms))]
    lambdas[0] = 1.0

    min = float('inf')
    # consider all possible combinations
    for curr_lambdas in product(*vals):
        if np.sum(curr_lambdas) != 1.0:
            continue
        # consider a combination of lambdas for calculate_prob_with_backoff and check its perplexity
        curr_perplexity = perplexity_backoff(dev_filename, lms, list(curr_lambdas), order=len(lms)-1)
        # update lambdas accordingly
        # want to minimize perplexity
        if curr_perplexity < min:
            min = curr_perplexity
            lambdas = list(curr_lambdas)
    return lambdas


# returns a list of labels
def label_cities(file_path, lms, countries, orders, interpolation=False, lambdas=None):
    labels = []
    with open(file_path, mode='r', errors='ignore') as f:
        for city in f:
            min = float("inf")
            idx = 0
            for j in range(0, len(lms)):
                probs_list = []
                N = 0
                i = 0
                while i < len(city):
                    full_n_gram = city[i:(orders[countries[j][:2]] + 1 + i)]
                    history = full_n_gram[:len(full_n_gram) - 1]
                    char = full_n_gram[-1]
                    if interpolation:
                        prob = calculate_prob_with_backoff(char, history, lms[j], lambdas[j])
                    else:
                        if history not in lms[j]:
                            history = '<UNK>'
                        probs_tuple = lms[j][history]
                        chars, probs = zip(*probs_tuple)
                        probIndex = chars.index(char)
                        prob = probs[probIndex]
                    probs_list.append(prob)
                    i += 1
                    N += 1
                if len(probs_list) == 0:
                    continue
                perplexity = np.sum(np.ma.log(probs_list)) * (-1 / N)
                if perplexity < min:
                    min = perplexity
                    idx = j
                    # print(idx, min)
            labels.append(countries[idx][:2])
    return labels


# Part 3
def text_classification(orders, add_k=1, interpolation=False, lambdas=None):
    train_path = "train/"
    val_path = "val/"
    test_file = "cities_test.txt"
    countries = ["af.txt", "cn.txt", "de.txt", "fi.txt", "fr.txt", "in.txt", "ir.txt", "pk.txt", "za.txt"]
    lms = []

    # currently assuming that P(Country) is the same for all countries
    # (all files contain same number of cities, so can't really do better)

    for country in countries:
        if interpolation:
            lms.append([train_char_lm(train_path + country, i, add_k=add_k) for i in range(0, orders[country[:2]] + 1)])
        else:
            lms.append(train_char_lm(train_path + country, orders[country[:2]], add_k=add_k))

    # validation
    val_labels = []
    y_true = []
    for country in countries:
        # for each city, guess what country it's in
        curr_labels = label_cities(val_path + country, lms, countries, orders, interpolation=interpolation, lambdas=lambdas)
        val_labels += curr_labels
        y_true += [country[:2] for i in range(0, len(curr_labels))]
        print(country, orders[country[:2]], sum([1 if lbl == country[:2] else 0 for lbl in curr_labels]) / len(curr_labels))
    print(sum([1 if val_labels[i] == y_true[i] else 0 for i in range(0, len(val_labels))]) / len(y_true))

    # test data
    test_labels = label_cities(test_file, lms, countries, orders, interpolation=interpolation, lambdas=lambdas)
    with open("labels.txt", mode='w') as out:
        out.write("\n".join(test_labels))


if __name__ == '__main__':
    print('Training language model')
    # lm = train_char_lm("shakespeare_input.txt", order=2, add_k=1)
    # print(generate_text(lm, 2))

    i = 4
    # print(set_lambdas([train_char_lm("train/af.txt", order=i) for i in range(0, i)], "val/af.txt"))
    # lambdas = []
    # lambdas.append(set_lambdas([train_char_lm("train/af.txt", order=i) for i in range(0, i)], "val/af.txt"))
    # lambdas.append(set_lambdas([train_char_lm("train/cn.txt", order=i) for i in range(0, i)], "val/cn.txt"))
    # lambdas.append(set_lambdas([train_char_lm("train/de.txt", order=i) for i in range(0, i)], "val/de.txt"))
    # lambdas.append(set_lambdas([train_char_lm("train/fi.txt", order=i) for i in range(0, i)], "val/fi.txt"))
    # lambdas.append(set_lambdas([train_char_lm("train/fr.txt", order=i) for i in range(0, i)], "val/fr.txt"))
    # lambdas.append(set_lambdas([train_char_lm("train/in.txt", order=i) for i in range(0, i)], "val/in.txt"))
    # lambdas.append(set_lambdas([train_char_lm("train/ir.txt", order=i) for i in range(0, i)], "val/ir.txt"))
    # lambdas.append(set_lambdas([train_char_lm("train/pk.txt", order=i) for i in range(0, i)], "val/pk.txt"))
    # lambdas.append(set_lambdas([train_char_lm("train/za.txt", order=i) for i in range(0, i)], "val/za.txt"))
    #
    # print(lambdas)

    lambdas3 = [[0.0, 0.9, 0.1, 0.0],
                [0.0, 0.9, 0.1, 0.0],
                [0.0, 0.9, 0.1, 0.0],
                [0.0, 0.9, 0.1, 0.0],
                [0.0, 0.7, 0.3, 0.0],
                [0.0, 0.9, 0.1, 0.0],
                [0.0, 0.9, 0.1, 0.0],
                [0.0, 0.7, 0.2, 0.1],
                [0.1, 0.9, 0.0, 0.0]]

    lambdas4 = [[0.5, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5, 0.0],
                [0.5, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5, 0.0, 0.0],
                [0.5, 0.0, 0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.5, 0.5, 0.0, 0.0]]

    lambdas5 = [[0.9, 0.0, 0.0, 0.0, 0.1, 0.0],
                [0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
                [0.0, 0.9, 0.0, 0.0, 0.1, 0.0],
                [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5, 0.0, 0.0, 0.0]]

    lambdas6 = [[0.1, 0.1, 0.1, 0.1, 0.0, 0.5, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.0, 0.5, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1]]

    lambdas7 = [[0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.2],
                [0.2, 0.2, 0.0, 0.0, 0.2, 0.2, 0.0, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.2],
                [0.2, 0.2, 0.0, 0.2, 0.2, 0.0, 0.0, 0.2],
                [0.0, 0.2, 0.0, 0.2, 0.2, 0.2, 0.0, 0.2],
                [0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2],
                [0.0, 0.2, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.0, 0.2, 0.2, 0.0, 0.0, 0.2]]

    n = 3
    orders = dict()
    orders['af'] = n
    orders['cn'] = n
    orders['de'] = n
    orders['fi'] = n
    orders['fr'] = n
    orders['in'] = n
    orders['ir'] = n
    orders['pk'] = n
    orders['za'] = n

    # lm = train_char_lm("shakespeare_input.txt", order=4, add_k=3)
    # lms = [train_char_lm("shakespeare_input.txt", order=i, add_k=3) for i in range(0, 5)]
    # print(perplexity_backoff("shakespeare_sonnets.txt", lms, order=4, lambdas=lambdas4[1]))
    # print(perplexity_backoff("nytimes_article.txt", lms, order=4, lambdas=lambdas4[1]))

    text_classification(orders, add_k=2)
    # text_classification(orders, add_k=1, interpolation=True, lambdas=lambdas3)