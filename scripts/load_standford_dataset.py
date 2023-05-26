


def load_sentences():
    f = open("C:/Users/Cerisara Nathan/Documents/GitHub/TIPE/data/standford_dataset/datasetSentences.txt", "r")
    lines = f.read().split("\n")[1:]
    f.close()
    #
    sentences = {}
    #
    for l in lines:
        ls = l.split("\t")
        if len(ls) < 2:
            continue
        index = int(ls[0])
        sentence = ls[1]
        #
        sentences[index] = sentence
    #
    return sentences


def split_dataset(sentences):
    f = open("C:/Users/Cerisara Nathan/Documents/GitHub/TIPE/data/standford_dataset/datasetSplit.txt", "r")
    lines = f.read().split("\n")[1:]
    f.close()
    #
    train = {}
    test = {}
    #
    for l in lines:
        ls = l.split(",")
        if len(ls) < 2:
            continue
        index = int(ls[0])
        classe = int(ls[1])
        #
        if index in sentences.keys():
            if classe == 1:
                train[index] = sentences[index]
            elif classe == 2:
                test[index] = sentences[index]
    #
    return train, test

def load_labels(sentences):
    f = open("C:/Users/Cerisara Nathan/Documents/GitHub/TIPE/data/standford_dataset/sentiment_labels.txt", "r")
    lines = f.read().split("\n")[1:]
    f.close()
    #
    labels = {}
    #
    for l in lines:
        ls = l.split("|")
        if len(ls) < 2:
            continue
        index = int(ls[0])
        label = float(ls[1])
        #
        if index in sentences.keys():
            labels[index] = label
    #
    return labels



def main_sdf_dataset_load():
    #print("Loading dataset...")
    sentences = load_sentences()
    train, test = split_dataset(sentences)
    print("TRAIN LENGTH : ", len(train), "\nTEST LENGTH : ", len(test))
    labels = load_labels(sentences)
    #
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    #
    # nb_max = 10000
    # i = 0
    for k in train.keys():
        X_train.append( train[k] )
        Y_train.append( labels[k] )
        # i += 1
        # if i >= nb_max:
        #     break
    #
    # nb_max = 10000
    # i = 0
    for k in test.keys():
        X_test.append( test[k] )
        Y_test.append( labels[k] )
        # i += 1
        # if i >= nb_max:
        #     break
    #
    return (X_train, Y_train), (X_test, Y_test)





