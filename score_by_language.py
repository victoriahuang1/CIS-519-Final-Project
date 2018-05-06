import score
import sys

def get_accuracies(y_pred, gold_labels):
    corrects = {}
    totals = {}
    for i in range(len(gold_labels)):
        if gold_labels[i] in totals:
            totals[gold_labels[i]] += 1
        else:
            totals[gold_labels[i]] = 1
            
        if y_pred[i] == gold_labels[i]:
            if gold_labels[i] in corrects:
                corrects[gold_labels[i]] += 1
            else:
                corrects[gold_labels[i]] = 1
    
    accuracies = {}
    for lang in totals:
        if lang in corrects:
            accuracies[lang] = corrects[lang] / totals[lang]
        else:
            accuracies[lang] = 0
    return accuracies
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Expecting paths to predictions file")
        sys.exit(1)

    y_pred, gold_labels = score.load_labels(sys.argv[1])
    accuracies = get_accuracies(y_pred, gold_labels)
    for lang in accuracies:
        print("Accuracy for " + lang + ": {}".format(accuracies[lang]))
