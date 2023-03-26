""" Importing libraries """

import pandas as pd
import numpy as np
import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
import os

""" Setting Variables """

ROOT_DIR = "../bert/"
TOKEN_SENTENCE_MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_LEVEL = 1 # 0 = print nothing, 1 = print low usage, 2 = print everything

""" Traite text function """

def traite_txt(txt):
    i = 0
    i = txt.find("@", i, len(txt))
    while i != -1 and i<len(txt)-1:
        if txt[i+1] != " ":
            j = txt.find(" ", i, len(txt))
            if j != -1:
                txt = txt[:i+1]+"user"+txt[j:]
                i = txt.find("@", j, len(txt))
            else:
                break
        else:
            i = txt.find("@", i+1, len(txt))
    #
    return txt


""" Tokenise string function """

def tokenize_text(text, tokenizer):
    if PRINT_LEVEL >= 2: print("Tokenizing ", text, type(text))
    textf = traite_txt(text)
    inputs = tokenizer.encode_plus(
        textf,
        None,
        pad_to_max_length=True,
        add_special_tokens=True,
        return_attention_mask=True,
        max_length=TOKEN_SENTENCE_MAX_LENGTH,
    )
    return inputs


""" Bert Dataset class """

class BertDataset(Dataset):
    def __init__(self, tokenizer, sentences, labels):
        super(BertDataset, self).__init__()
        #
        self.sentences = sentences # liste d'éléments
        self.labels = labels # liste de labels
        #
        self.tokenizer=tokenizer
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        
        sentence = self.sentences[index]
        label = self.labels[index]
        
        inputs = tokenize_text(sentence, self.tokenizer)
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': torch.tensor(label, dtype=torch.long)
        }

""" Bert class """


class FF(nn.Module):
    def __init__(self, nb_classes):
        super(FF, self).__init__()
        L = [768, 768*4, 768*8, 768*8, 768*4]
        L.append(int((L[-1]+nb_classes)/2))
        self.lin1 = nn.Linear(L[0], L[1])
        self.lin2 = nn.Linear(L[1], L[2])
        self.lin3 = nn.Linear(L[2], L[3])
        self.relu1 = nn.ReLU()
        self.lin4 = nn.Linear(L[3], L[4])
        self.lin5 = nn.Linear(L[4], L[5])
        self.relu2 = nn.ReLU()
        self.lin6 = nn.Linear(L[5], nb_classes)
        #
        self.lst = [self.lin1, self.lin2, self.lin3, self.relu1, self.lin4, self.lin5, self.relu2, self.lin6]
    
    def forward(self, x):
        for f in self.lst:
            x = f(x)
        return x


class BERT(nn.Module):
    def __init__(self, nb_classes=1):
        super(BERT, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        # self.out = nn.Linear(768, nb_classes)
        self.out = FF(nb_classes)
        
    def forward(self,ids,mask,token_type_ids):
        _,o2= self.bert_model(ids,attention_mask=mask,token_type_ids=token_type_ids, return_dict=False)
        
        out= self.out(o2)
        
        return out


""" Finetune function """


def finetune(epochs,dataloader,model,loss_fn,optimizer):
    if PRINT_LEVEL >= 1: print("Starting of finetune function")
    model.to(DEVICE)
    model.train()
    for  epoch in range(epochs):
        print("epoch : ", epoch)
        
        loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))

        for batch, dl in loop:
            ids=dl['ids'].to(DEVICE)
            token_type_ids=dl['token_type_ids'].to(DEVICE)
            mask=dl['mask'].to(DEVICE)
            label=dl['target'].to(DEVICE)
            label = label.unsqueeze(1)
            
            optimizer.zero_grad()
            
            output=model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids)
            label = label.type_as(output)

            loss=loss_fn(output,label)
            loss.backward()
            
            optimizer.step()
            
            pred = torch.where(output >= 0, 1, 0)

            num_correct = sum(1 for a, b in zip(pred, label) if a[0] == b[0])
            num_samples = pred.shape[0]
            accuracy = num_correct/num_samples
            
            print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
            
            # Show progress while training
            loop.set_description(f'Epoch={epoch}/{epochs}')
            loop.set_postfix(loss=loss.item(),acc=accuracy)

    return model

""" Loading tokenizer """

def load_tokenizer():
    if PRINT_LEVEL >= 1: print("Loading tokenizer...")
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer

""" Loading datasets """
def load_datasets(tokenizer, csv_path, nb_classes, csv_del):
    if PRINT_LEVEL >= 1: print("Loading datasets ", csv_path, " ...")
    dataset= BertDataset(tokenizer, csv_path=csv_path, csv_delimiter=csv_del)
    dataloader=DataLoader(dataset=dataset,batch_size=32)
    return dataset, dataloader


""" Saving/Loading model function """


def save_model(path, model, optimizer, nb_classes):
    if PRINT_LEVEL >= 1: print("Saving model checkpoints ", path, " ...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'nb_classes': nb_classes
        }, path)


def init_model(path, nb_classes):
    if PRINT_LEVEL >= 1: print(f"Initializing the model (nb_classes={nb_classes})")
    # BERT model
    model=BERT(nb_classes)
    # loss function
    loss_fn = nn.BCEWithLogitsLoss()

    #Initialize Optimizer
    optimizer= optim.Adam(model.parameters(),lr= 0.0001)

    for param in model.bert_model.parameters():
        param.requires_grad = False
    
    return model, loss_fn, optimizer


def load_model(path): # On suppose que le fichier existe
    if PRINT_LEVEL >= 1: print("Loading model from ", path, " ...")
    t = torch.load(path)
    # BERT model
    model=BERT(t['nb_classes'])
    # loss function
    loss_fn = nn.BCEWithLogitsLoss()

    #Initialize Optimizer
    optimizer= optim.Adam(model.parameters(),lr= 0.0001)

    for param in model.bert_model.parameters():
        param.requires_grad = False

    model.load_state_dict(t['model_state_dict'])
    optimizer.load_state_dict(t['optimizer_state_dict'])
    model.eval()

    return model, loss_fn, optimizer





