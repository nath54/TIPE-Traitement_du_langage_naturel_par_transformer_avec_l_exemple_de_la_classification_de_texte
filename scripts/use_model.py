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

print("Loading my BERT custom classifier model...")
classifier = ClassifierFF()
experience_model = Experience("standford_experience_model4", classifier, "use")

q = input(">>>")
while q != "q":
    tensor = experience_model.use_model(q)
    score = tensor.item()
    print("Score: ", score)
    # clear CUDA memory
    del tensor
    torch.cuda.empty_cache()
    #    
    q = input(">>>")
