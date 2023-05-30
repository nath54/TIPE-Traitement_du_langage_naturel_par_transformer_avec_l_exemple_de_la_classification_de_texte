print("Importing libraries...")
from pysentimiento import create_analyzer

from tqdm import tqdm
import json

import numpy as np
import torch

torch.set_grad_enabled(False)

f = open("themes.json", "r")
themes = json.load(f)
f.close()

print("Loading the dataset file...")

f = open("D:/datasets/news.json")
lines = f.readlines()
f.close()

msgs = [json.loads(l)["content"].lower() for l in lines]

themes_msgs = {}
for t in themes:
    themes_msgs[t] = []

print("Finding themes in the messages...")

used = {}

for t in themes:
    n = 0
    for m in tqdm(msgs):
        dedans = False
        for tt in themes[t]:
            ttt = tt.lower()
            if ttt in m:
                if not ttt in used:
                    used[ttt] = 1
                else:
                    used[ttt] += 1
                #
                dedans = True
                break
        if dedans : 
            themes_msgs[t].append(m)
            n+=1

f = open("debug_used.json", "w")
json.dump(used, f)
f.close()

#

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
            res = [(res[itm], itm) for itm in res.keys()]
            res.sort()
            #
            sentiments[m] = res
        #
        if res[2][1] == "POS": nb_pos+=1
        elif res[2][1] == "NEU": nb_neu+=1
        elif res[2][1] == "NEG": nb_neg+=1

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

