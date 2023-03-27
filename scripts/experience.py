""" Importing libraries """

print("Importing the libraries...")
import numpy as np
import matplotlib.pyplot as plt
import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
import os
from datetime import datetime
from math import *
from torch.utils.tensorboard import SummaryWriter


""" Dataset """

class ExpDataset(Dataset):
    def __init__(self, tokenizer, data, experience):
        super(ExpDataset, self).__init__()
        self.experience = experience
        #
        self.X = data[0]
        self.Y = data[1]
        #
        self.tokenizer=tokenizer
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        
        sentence = self.X[index]
        label = self.Y[index]
        
        inputs = self.experience.tokenize_text(sentence, self.tokenizer)
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': torch.tensor(label, dtype=torch.float32)
        }




""" Bert Model """

class FullModelBertClassifier(nn.Module):
    def __init__(self, classifier_model):
        super(FullModelBertClassifier, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.classifier = classifier_model
        
    def forward(self,ids,mask,token_type_ids):
        _, out= self.bert_model(ids,attention_mask=mask,token_type_ids=token_type_ids, return_dict=False)

        out = self.classifier(out)

        return out


""" Objet experience """

class Experience:
    def __init__(self, model_name, train, test, classifier_model):
        #
        self.root_dir = "C:/Users/Cerisara Nathan/Documents/GitHub/TIPE/"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.token_sentence_max_length = 512
        print("Device : ", self.device)
        #
        self.model_name = model_name
        print("Loading the model...")
        self.model = FullModelBertClassifier(classifier_model).to(self.device)
        print("Loading the tokenizer...")
        self.tokenizer = self.load_tokenizer()
        print("Loading the loss function and the optimizer...")
        self.loss_fn = nn.MSELoss().to(self.device)
        self.optimizer= optim.Adam(self.model.parameters(),lr= 0.00001)
        #
        self.train_dataset = ExpDataset(self.tokenizer, train, self)
        self.test_dataset = ExpDataset(self.tokenizer, test, self)
        #

        #
        for param in self.model.bert_model.parameters():
            param.requires_grad = False # On n'entrainera pas les paramètres de BERT
        #
        pth = self.root_dir + "torch_saves/"+model_name+"_state.pt"
        if os.path.exists(pth):
            print("An existing weights state has been found, loading it...")
            self.load_model_state(pth)
    
    # Loading tokenizer
    def load_tokenizer(self):
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
        return tokenizer

    # Tokenize text
    def tokenize_text(self, text, tokenizer):
        inputs = tokenizer.encode_plus(
            text,
            None,
            padding="max_length",
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.token_sentence_max_length,
            truncation=True
        )
        return inputs

    # Load model state
    def load_model_state(self, path):
        #
        t = torch.load(path)
        #
        self.model.load_state_dict(t['model_state_dict'])
        self.optimizer.load_state_dict(t['optimizer_state_dict'])
        self.model.eval()

    # Save model state
    def save_model_state(self, path):
        #
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    # Function to use the model
    def use_model(self, text_input):
        #
        tt = self.tokenize_text(text_input, self.tokenizer)

        ids=torch.tensor([tt['input_ids']]).to(self.device)
        token_type_ids=torch.tensor([tt['token_type_ids']]).to(self.device)
        mask=torch.tensor([tt['attention_mask']]).to(self.device)

        output=self.model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids)

        return output

    # Train the model
    def train_model(self, epochs):
        #
        tb = SummaryWriter()
        #
        data_sampler = RandomSampler(self.train_dataset, num_samples=100)
        dataloader = DataLoader(self.train_dataset, 16, sampler=data_sampler)
        test_sampler = RandomSampler(self.test_dataset, num_samples=50)
        testloader = DataLoader(self.test_dataset, 16, sampler=test_sampler)
        #
        last_opt_cg = 0
        #
        print("Preparing the model to train...")
        self.model.to(self.device)
        self.model.train()
        #
        for epoch in range(epochs):
            
            dmoys_epoch = []
            losses_epoch = []
            accuracy_epoch = []

            print("epoch : ", epoch)
            
            loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))

            for batch, dl in loop:
                ids = dl['ids'].to(self.device)
                token_type_ids = dl['token_type_ids'].to(self.device)
                mask = dl['mask'].to(self.device)
                label = dl['target'].to(self.device)
                label = label.unsqueeze(1)

                self.optimizer.zero_grad()
                
                output = self.model(
                    ids=ids,
                    mask=mask,
                    token_type_ids=token_type_ids).to(self.device)
                label = label.type_as(output)

                loss = self.loss_fn(output,label)
                loss.backward()
                losses_epoch.append(loss.item())
                
                self.optimizer.step()

                dists = [abs(a[0]-b[0]) for a, b in zip(output, label)]
                dmoy = sum(dists)/len(dists)
                dmoys_epoch.append(dmoy)
                
                num_correct = sum(1 for a, b in zip(output, label) if abs(a[0]-b[0]) <= 0.1 )
                num_samples = output.shape[0]
                accuracy = num_correct/num_samples

                accuracy_epoch.append(accuracy)
                
                print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}, (dist moy = {dmoy})')
                
                # Show progress while training
                loop.set_description(f'Epoch={epoch}/{epochs}')
                loop.set_postfix(loss=loss.item(),acc=accuracy)

            
            tb.add_scalar("Loss", sum(losses_epoch)/len(losses_epoch), epoch)
            tb.add_scalar("Accuracy", sum(accuracy_epoch)/len(accuracy_epoch), epoch)
            tb.add_scalar("Distance Moy", sum(dmoys_epoch)/len(dmoys_epoch), epoch)

            # for name, weight in self.model.classifier.named_parameters():
            #     tb.add_histogram(name,weight, epoch)
            #     tb.add_histogram(f'{name}.grad',weight.grad, epoch)

            self.save_model_state(self.root_dir + "torch_saves/"+self.model_name+"_state.pt")

            # On va tester le modèle sur le test dataset

            
            dmoys_epoch_test = []
            losses_epoch_test = []
            accuracy_epoch_test = []
            loop_test=tqdm(enumerate(testloader),leave=False,total=len(testloader))
            for batch, dl in loop_test:
                ids = dl['ids'].to(self.device)
                token_type_ids = dl['token_type_ids'].to(self.device)
                mask = dl['mask'].to(self.device)
                label = dl['target'].to(self.device)
                label = label.unsqueeze(1)
                
                output = self.model(
                    ids=ids,
                    mask=mask,
                    token_type_ids=token_type_ids)
                label = label.type_as(output)

                loss = self.loss_fn(output,label)
                loss.backward()
                losses_epoch_test.append(loss.item())

                dists = [abs(a[0]-b[0]) for a, b in zip(output, label)]
                dmoy = sum(dists)/len(dists)
                dmoys_epoch_test.append(dmoy)
                
                num_correct = sum(1 for a, b in zip(output, label) if abs(a[0]-b[0]) <= 0.1 )
                num_samples = output.shape[0]
                accuracy = num_correct/num_samples

                accuracy_epoch_test.append(accuracy)
                
                print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}, (dist moy = {dmoy})')
                
                # Show progress while training
                loop_test.set_description(f'Epoch={epoch}/{epochs}')
                loop_test.set_postfix(loss=loss.item(),acc=accuracy)


            
            tb.add_scalar("Loss Test", sum(losses_epoch_test)/len(losses_epoch_test), epoch)
            tb.add_scalar("Accuracy Test", sum(accuracy_epoch_test)/len(accuracy_epoch_test), epoch)
            tb.add_scalar("Distance Moy Test", sum(dmoys_epoch_test)/len(dmoys_epoch_test), epoch)


            # Update optimizer learning rate if needed

            if last_opt_cg >= 10:
                mll = sum(losses_epoch[-5:])/5
                dlm = [abs(mll-losses_epoch[-i]) for i in range(1, 6)]
                dlm_moy = sum(dlm)/5
                if dlm_moy < 0.001:
                    for g in self.optimizer.param_groups:
                        g['lr'] = g['lr']/2.0
                    # 
                    print("\n\nLearning rate changed to "+str(self.optimizer.param_groups[0]['lr'])+" !!!")

                else:
                    last_opt_cg += 1
            else:
                last_opt_cg += 1




