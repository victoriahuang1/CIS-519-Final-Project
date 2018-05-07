import sys
import language_model


# Loads the labels from file into array
def load_labels(filepath, pred_fam=False):
    y_pred = []
    gold_labels = []
    with open(filepath, mode='r', errors='ignore') as f:
        for line in f:
            words = line.split()
            pred = words[0].strip().lower()
            gold = words[1].strip().lower()
            if pred_fam:
                pred = language_model.lang_to_fam[pred]
                gold = language_model.lang_to_fam[gold]
            y_pred.append(pred)
            gold_labels.append(gold)
    f.close()
    return y_pred, gold_labels


# Computes accuracy of model
def get_accuracy(y_pred, labels):
    return sum([1 if y_pred[i] == labels[i] else 0 for i in range(len(labels))]) / float(len(labels))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Expecting paths to predictions file")
        sys.exit(1)

    y_pred, gold_labels = load_labels(sys.argv[1], pred_fam=True)

    print("Accuracy: {}".format(get_accuracy(y_pred, gold_labels)))
