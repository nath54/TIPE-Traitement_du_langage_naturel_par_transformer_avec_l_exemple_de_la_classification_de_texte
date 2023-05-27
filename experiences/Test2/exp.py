
import sys
sys.path.insert(1, "../../scripts/")
from experience import *
from load_standford_dataset import *


""" ClassifierFF Model Dev Version """

class ClassifierFF(nn.Module):
    def __init__(self): # 768 -> 1
        super(ClassifierFF, self).__init__()
        #
        # Linear function
        self.lin1 = nn.Linear(768, 1536)
        self.lin2 = nn.Linear(1536, 3072)
        self.lin3 = nn.Linear(3072, 3072)
        self.lin4 = nn.Linear(3072, 384)
        # Linear function (readout)
        self.lin5 = nn.Linear(384, 1)  
        # Non-linearity
        self.nl1 = nn.Softmax()
        #
        self.lst = [self.lin1, self.lin2, self.lin3, self.lin4, self.lin5, self.nl1]
    
    def forward(self, x):
        for f in self.lst:
            x = f(x)
        return x



""" Main Code """

if __name__ == "__main__":

    classifier_model = ClassifierFF()

    train, test  = main_sdf_dataset_load()
    print(f"Train : {len(train[0])}, Test : {len(test[0])}")

    experience = Experience("test1", classifier_model, "train", train, test)

    experience.train_model(100)

    # experience.save_model_state()

