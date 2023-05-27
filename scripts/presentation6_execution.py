print("Importing libraries...")
import sys
sys.path.insert(1, "../experiences/StandfordExperience/")
from standford_experience import ClassifierFF
from experience import Experience

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


print("Loading my BERT custom classifier model...")
classifier = ClassifierFF()
experience_model = Experience("standford_experience_model4", classifier, "use")

all_diff_results = set()

print("Calculating the sentiment of each messages with the themes...")
for t in themes:
    scores = []
    for m in tqdm(themes_msgs[t]):
        if m in sentiments:
            score = sentiments[m]
        else:
            tensor = experience_model.use_model(m)
            score = tensor.item()
            
            all_diff_results.add(score)
            
            # clear CUDA memory
            del tensor
            torch.cuda.empty_cache()

            #
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
        "number_of_messages": len(scores),
        "average": average,
        "median": median,
        "variance": variance,
        "standard deviation": std_dev,
        "min": min_val,
        "max": max_val
    }

print("Saving the final results...")

f=open("final_results.json", "w")
json.dump(themes_scores, f)
f.close()

