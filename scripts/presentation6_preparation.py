
""" File: `application_preparation.py`"""


print("Importing libraries...")
from tqdm import tqdm
import json

import gensim.downloader as api
from gensim.models import Word2Vec

print("Loading Word2Vec model...")
model = api.load('word2vec-google-news-300')


themes = {
    "city": [],
    "cars": [],
    "house": [],
    "plants": [],
    "stores": [],
    "police": [],
    "hospitals": []
}

print("Calculating the themes vocabulary...")

for t in tqdm(themes):
    themes[t] = [s[0] for s in model.most_similar(t, topn=50)]

f = open("themes.json","w")
json.dump(themes, f)
f.close()