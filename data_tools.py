import os


# Parses files of the format `<language>, <family>` to map languages to their language families
def read_lang_to_fam(filepath):
    lang_to_fam = {}
    with open(filepath, mode='r', errors='ignore', encoding='utf-8') as f:
        for line in f:
            parsed_line = line.split(',')
            lang_to_fam[parsed_line[0].strip().lower()] = parsed_line[1].strip().lower()
    f.close()
    return lang_to_fam


# Reads a wiki file and returns a list of documents
def read_wiki_file(datapath):
    docs = []

    with open(datapath, mode='r', errors='ignore', encoding='utf-8') as f:
        curr_doc = []
        for line in f:
            if '</doc>' in line:
                docs.append(curr_doc)
                curr_doc = []
                continue
            elif '<doc' in line:
                continue
            words = line.split()
            curr_doc += words
    f.close()
    if len(curr_doc) > 0:
        docs.append(curr_doc)
    return docs


# Writes all characters in train and test set to file
def get_all_chars(train_dir, test_dir, out_path):
    chars = {}

    for subdir in os.listdir(train_dir):
        chars[subdir] = set()
        for doc in os.listdir(train_dir + subdir):
            with open(train_dir + subdir + '/' + doc, mode='r', errors='ignore', encoding='utf-8') as f:
                for line in f:
                    for c in line:
                        chars[subdir].add(c)
            f.close()

    for subdir in os.listdir(test_dir):
        for doc in os.listdir(test_dir + subdir):
            with open(test_dir + subdir + '/' + doc, mode='r', errors='ignore', encoding='utf-8') as f:
                for line in f:
                    for c in line:
                        chars[subdir].add(c)
            f.close()

    for subdir, vals in chars.items():
        with open(out_path + "_" + subdir, 'w', encoding='utf-8') as out:
            for c in vals:
                out.write("{}\n".format(c))


# Reads and returns a set of all characters in train and test sets
def load_all_chars(char_file):
    chars = set()
    with open(char_file, mode='r', errors='ignore', encoding='utf-8') as f:
        for line in f:
            for c in line:
                chars.add(c)
    f.close()
    return chars


# Loads examples (lists of lists of strings) and their labels
def load_wiki_data(data_dir):
    examples = []
    labels = []
    for subdir in os.listdir(data_dir):
        for doc in os.listdir(data_dir + subdir):
            new_docs = read_wiki_file(data_dir + subdir + '/' + doc)
            for i in range(len(new_docs)):
                labels.append(subdir)
            examples += new_docs
    return examples, labels

def load_gutenberg_data(data_dir):
    examples = []
    labels = []
    for subdir in os.listdir(data_dir):
        for doc in os.listdir(data_dir + subdir):
            examples.append(doc)
            labels.append(subdir)
    return examples, labels

# Writes predictions and gold labels to file
def write_pred(y_pred, gold_labels, out_path):
    with open(out_path, 'w') as out:
        for i in range(0, len(y_pred)):
            out.write("{}\t{}\n".format(y_pred[i], gold_labels[i]))
    out.close()