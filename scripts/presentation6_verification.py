print("Importing libraries...")
from pysentimiento import create_analyzer

from tqdm import tqdm
import json

import numpy as np
import torch

torch.set_grad_enabled(False)

f = open("final_results.json", "r")
themes = json.load(f)
f.close()

print("Loading the dataset file...")

f = open("D:/datasets/news.json")
lines = f.readlines()
f.close()

msgs = [json.loads(l)["content"] for l in lines]

themes_msgs = {}
for t in themes:
    themes_msgs[t] = set()

print("Finding themes in the messages...")

for m in tqdm(msgs):
    for t in themes:
        in_ = False
        for tt in themes[t]:
            if tt in m:
                in_ = True
                break
        if in_ : 
            themes_msgs[t].add(m)

sentiments = {}

themes_scores = {}


print("Loading classifier model...")
analyzer = create_analyzer(task="sentiment", lang="en")

print("Calculating the sentiment of each messages with the themes...")
for t in themes:
    scores = []
    nb_pos = 0
    nb_neu = 0
    nb_neg = 0
    for m in tqdm(themes_msgs[t]):
        if m in sentiments:
            res = sentiments[m]
        else:
            res = analyzer.predict(m).probas
            res = [(res[itm], itm) for itm in res.keys]
            res.sort()
            #
            sentiments[m] = res
        #
        if res[0][1] == "POS": nb_pos+=1
        elif res[0][1] == "NEU": nb_neu+=1
        elif res[0][1] == "NEG": nb_neg+=1

    #
    themes_scores[t] = {
        "positive": nb_pos,
        "neutral": nb_neu,
        "negative": nb_neg
    }

print("Saving the final results...")

f=open("verification_results.json", "w")
json.dump(themes_scores, f)
f.close()

