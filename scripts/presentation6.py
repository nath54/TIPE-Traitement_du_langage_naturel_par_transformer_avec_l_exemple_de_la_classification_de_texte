print("Importing libraries...")
import sys
sys.path.insert(1, "../experiences/StandfordExperience/")
from standford_experience import ClassifierFF
from experience import Experience

from tqdm import tqdm
import json

import gensim.downloader as api
from gensim.models import Word2Vec

import numpy as np

print("Loading my BERT custom classifier model...")
classifier = ClassifierFF()
experience_model = Experience("standford_experience_model3", classifier, "use")


print("Loading Word2Vec model...")
model = api.load('word2vec-google-news-300')


themes = {
    "city": [],
    "cars": [],
    "house": [],
    "apparts": [],
    "stores": [],
    "police": [],
    "hospitals": []
}

print("Calculating the themes vocabulary...")

for t in tqdm(themes):
    themes[t] = [s[0] for s in model.most_similar(t, topn=40)]

print("Loading the dataset file...")

f = open("D:/datasets/news.json")
lines = f.readlines()
f.close()

msgs = [json.loads(l)["content"] for l in lines]

themes_msgs = {}
for t in themes:
    themes_msgs[t] = []

print("Finding themes in the messages...")

for m in tqdm(msgs):
    for t in themes:
        in_ = False
        for tt in themes[t]:
            if tt in m:
                in_ = True
                break
        if in_ : 
            themes_msgs[t].append(m)

sentiments = {}

themes_scores = {}


print("Calculating the sentiment of each messages with the themes...")
for t in themes:
    scores = []
    for m in tqdm(themes_msgs[t]):
        if m in sentiments:
            score = sentiments[m]
        else:
            score = experience_model.use_model(m)
            sentiments[m] = score
        #
        scores.append(score)
    #
    average = np.mean(scores)
    median = np.median(scores)
    variance = np.var(scores)
    std_dev = np.std(scores)
    min_val = np.min(scores)
    max_val = np.max(scores)
    #
    themes_scores[t] = {
        "average": average,
        "median": median,
        "variance": variance,
        "standard deviation": std_dev,
        "min": min_val,
        "max": max_val
    }

print("Saving the final results...")

f=open("final_results.json")
json.dump(themes_scores, f)
f.close()

