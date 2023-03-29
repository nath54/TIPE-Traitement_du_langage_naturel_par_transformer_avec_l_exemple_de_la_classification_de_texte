

def load_sentences():
    f = open("C:/Users/Cerisara Nathan/Documents/Github/TIPE/data/twitter_1.6/training.1600000.processed.noemoticon.nopseudo.csv", "r")
    lines = f.read().split("\n")
    f.close()

    posit = []
    negat = []

    for l in lines:
        ls = l.split(",")
        label = ls[0]
        sentence = ",".join(ls[5:])[1:-1]

        # print(label)
        
        if label == '"0"':
            negat.append(sentence)
        elif label == '"4"':
            posit.append(sentence)
    
    lpos = len(posit)
    lneg = len(negat)

    # print("posit : ", len(posit))
    # print("negat : ", len(negat))

    rat = 0.2

    l1 = int(lpos*(1-rat))
    l2 = int(lneg*(1-rat))

    txt_train_posit = "\n".join(posit[0: l1])
    txt_train_negat = "\n".join(negat[0: l2])
    txt_teste_posit = "\n".join(posit[l1:])
    txt_teste_negat = "\n".join(negat[l2:])

    f = open("C:/Users/Cerisara Nathan/Documents/GitHub/TIPE/data/twitter_1.6/train_posit.txt", "w")
    f.write(txt_train_posit)
    f.close()

    
    f = open("C:/Users/Cerisara Nathan/Documents/GitHub/TIPE/data/twitter_1.6/train_negat.txt", "w")
    f.write(txt_train_negat)
    f.close()

    
    f = open("C:/Users/Cerisara Nathan/Documents/GitHub/TIPE/data/twitter_1.6/teste_posit.txt", "w")
    f.write(txt_teste_posit)
    f.close()

    
    f = open("C:/Users/Cerisara Nathan/Documents/GitHub/TIPE/data/twitter_1.6/teste_negat.txt", "w")
    f.write(txt_teste_negat)
    f.close()



load_sentences()

