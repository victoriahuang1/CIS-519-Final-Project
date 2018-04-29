import sys


# Loads the labels from file into array
def load_labels(filepath):
    y_pred = []
    gold_labels = []
    with open(filepath, mode='r', errors='ignore') as f:
        for line in f:
            words = line.split()
            y_pred.append(words[0].strip().lower())
            gold_labels.append(words[1].strip().lower())
    f.close()
    return y_pred, gold_labels


# Computes accuracy of model
def get_accuracy(y_pred, labels):
    return sum([1 if y_pred[i] == labels[i] else 0 for i in range(len(labels))]) / float(len(labels))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Expecting paths to predictions file")
        sys.exit(1)

    y_pred, gold_labels = load_labels(sys.argv[1])

    print("Accuracy: {}".format(get_accuracy(y_pred, gold_labels)))
