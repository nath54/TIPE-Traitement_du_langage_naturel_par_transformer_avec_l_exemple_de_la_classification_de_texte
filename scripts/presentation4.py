import torch
from transformers import BertModel
from transformers import BertTokenizer
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertModel.from_pretrained("bert-base-uncased").cuda()


def get_embedding(word):
    with torch.no_grad():
        tokens = tokenizer.tokenize(word)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_ids = torch.tensor([token_ids]).cuda()
        outputs = model(input_ids)
        word_embedding = outputs.last_hidden_state[0][0]
        #
        del tokens
        del token_ids
        del input_ids
        del outputs
        torch.cuda.empty_cache()
        return word_embedding

f = open("C:/Users/Cerisara Nathan/Documents/Github/TIPE/data/words_2.txt", "r")
words = f.readlines()
f.close()

q = input("Enter a word : ")
while q != "q":
    e_q = get_embedding(q)
    #
    dists_w = []
    i = 0
    for w in tqdm(words):
        e_w = get_embedding(w.strip())  # strip off the newline characters
        d = torch.dist(e_q, e_w)
        dists_w.append((d.item(), w))  # convert tensor to a simple Python number
        del e_w
        torch.cuda.empty_cache()
        #
    #
    dists_w.sort()
    for i in range(30):
        print(f"{i}) {dists_w[i][1]} - {dists_w[i][0]}")
    #
    q = input("Enter a word : ")
