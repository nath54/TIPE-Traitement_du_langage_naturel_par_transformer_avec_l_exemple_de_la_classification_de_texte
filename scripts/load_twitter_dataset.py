
""" File: `load_twitter_dataset.py`"""

import random
import torch

def load_sentences(path):
    f = open(path, "r")
    sentences = f.readlines()
    f.close()
    return sentences

def tensor_with_1(n, i):
    t = torch.zeros(n)
    t[i] = 1.0
    return t

def main_twt_dataset_load():
    #print("Loading dataset...")
    N = 50000
    train_pos = load_sentences("C:/Users/Cerisara Nathan/Documents/GitHub/TIPE/data/twitter_1.6/train_posit.txt")
    train_pos = train_pos[:min(len(train_pos), N)]
    train_neg = load_sentences("C:/Users/Cerisara Nathan/Documents/GitHub/TIPE/data/twitter_1.6/train_negat.txt")[:len(train_pos)]
    test_pos = load_sentences("C:/Users/Cerisara Nathan/Documents/GitHub/TIPE/data/twitter_1.6/teste_posit.txt")[:N//2]
    test_neg = load_sentences("C:/Users/Cerisara Nathan/Documents/GitHub/TIPE/data/twitter_1.6/teste_negat.txt")[:N//2]
    train = [(t, tensor_with_1(2, 1)) for t in train_pos] + [(t, tensor_with_1(2, 0)) for t in train_neg]
    test = [(t, tensor_with_1(2, 1)) for t in test_pos] + [(t, tensor_with_1(2, 0)) for t in test_neg]
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
