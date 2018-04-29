import os


# Parses files of the format `<language>, <family>` to map languages to their language families
def read_lang_to_fam(filepath):
    lang_to_fam = {}
    with open(filepath, mode='r', errors='ignore') as f:
        for line in f:
            parsed_line = line.split(',')
            lang_to_fam[parsed_line[0].strip().lower()] = parsed_line[1].strip().lower()
    f.close()
    return lang_to_fam


# Reads a wiki file and returns a list of documents
def read_wiki_file(datapath):
    docs = []

    with open(datapath, mode='r', errors='ignore') as f:
        curr_doc = []
        for line in f:
            if '</doc>' in line or '<doc' in line:
                curr_doc = []
                docs.append(curr_doc)
                continue
            words = line.split()
            curr_doc += words
    f.close()
    docs.append(curr_doc)
    return docs


# Loads examples and their labels
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