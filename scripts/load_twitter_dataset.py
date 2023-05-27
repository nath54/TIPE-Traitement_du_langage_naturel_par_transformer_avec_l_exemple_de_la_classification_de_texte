import random

def load_sentences(path):
    f = open(path, "r")
    sentences = f.readlines()
    f.close()
    return sentences

def main_twt_dataset_load():
    #print("Loading dataset...")
    train_pos = load_sentences("C:/Users/Cerisara Nathan/Documents/GitHub/TIPE/data/twitter_1.6/train_posit.txt")
    train_neg = load_sentences("C:/Users/Cerisara Nathan/Documents/GitHub/TIPE/data/twitter_1.6/train_negat.txt")[:len(train_pos)]
    test_pos = load_sentences("C:/Users/Cerisara Nathan/Documents/GitHub/TIPE/data/twitter_1.6/teste_posit.txt")
    test_neg = load_sentences("C:/Users/Cerisara Nathan/Documents/GitHub/TIPE/data/twitter_1.6/teste_negat.txt")
    train = [(t, 1) for t in train_pos] + [(t, 0) for t in train_neg]
    test = [(t, 1) for t in test_pos] + [(t, 0) for t in test_neg]
    random.shuffle(train)
    random.shuffle(test)
    #
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    #
    # nb_max = 10000
    # i = 0
    for t in train:
        X_train.append( t[0] )
        Y_train.append( t[1] )
        # i += 1
        # if i >= nb_max:
        #     break
    #
    # nb_max = 10000
    # i = 0
    for t in test:
        X_test.append( t[0] )
        Y_test.append( t[1] )
        # i += 1
        # if i >= nb_max:
        #     break
    #
    return (X_train, Y_train), (X_test, Y_test)
