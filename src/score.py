import sys


# Loads the labels from file into array
def load_labels(filepath):
    labels = []
    with open(filepath, mode='r', errors='ignore') as f:
        for line in f:
            labels.append(line.strip())
    f.close()
    return labels


# Computes accuracy of model
def get_accuracy(y_pred, labels):
    return sum([1 if y_pred[i] == labels[i] else 0 for i in range(len(labels))]) / float(len(labels))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Expecting paths to predictions and gold labels files")
        sys.exit(1)

    y_pred = load_labels(sys.argv[1])
    # TODO change to infer labels from directories?
    labels = load_labels(sys.argv[2])

    print("Accuracy: {}".format(get_accuracy(y_pred, labels)))
