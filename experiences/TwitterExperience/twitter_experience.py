
import sys
sys.path.insert(1, "../../scripts/")
from experience import *
from load_twitter_dataset import *


""" ClassifierFF Model Dev Version """

# class ClassifierFF(nn.Module):
#     def __init__(self): # 768 -> 1
#         super(ClassifierFF, self).__init__()
#         #
#         # Linear function
#         self.lin1 = nn.Linear(768, 768*5)
#         self.lin2 = nn.Linear(768*5, 768*15)
#         self.lin3 = nn.Linear(768*15, 768*5)
#         self.lin4 = nn.Linear(768*5, 384)
#         # Non-linearity
#         self.nl1 = nn.Softmax()
#         # Linear function (readout)
#         self.lin5 = nn.Linear(384, 1)  
#         #
#         self.lst = [self.lin1, self.lin2, self.lin3, self.lin4, self.nl1, self.lin5]
    
#     def forward(self, x):
#         for f in self.lst:
#             x = f(x)
#         return x
    

class ClassifierFF(nn.Module):
    def __init__(self): # 768 -> 1
        super(ClassifierFF, self).__init__()
        #
        # Linear function
        self.lin1 = nn.Linear(768, 7680*2)
#         # Non-linearity
        self.nl1 = nn.Softmax()
        # Linear function (readout)
        self.lin2 = nn.Linear(7680*2, 1)  
        #
        self.lst = [self.lin1, self.nl1, self.lin2]
    
    def forward(self, x):
        for f in self.lst:
            x = f(x)
        return x



""" Main Code """

if __name__ == "__main__":
    classifier_model = ClassifierFF()

    train, test  = main_twt_dataset_load()
    print(f"Train : {len(train[0])}, Test : {len(test[0])}")

    experience = Experience("twitter_experience_model2", classifier_model, "train", train, test)

    experience.train_model(150)

    # experience.save_model_state()

