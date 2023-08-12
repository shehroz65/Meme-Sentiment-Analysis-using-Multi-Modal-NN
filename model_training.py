#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from PIL import Image
import spacy
import nltk
import numpy as np
import re
import torch.nn.functional as F
import torch.optim as optim
import spacy
import torch
from PIL import Image


import os
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# In[2]:


get_ipython().system('pip install -U pip setuptools wheel')
get_ipython().system('pip install -U spacy')


# In[3]:


import spacy

spacy.prefer_gpu()
spacy_eng = spacy.load("en_core_web_sm")


# In[4]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


# In[5]:


ref = pd.read_csv("reference.csv")


# In[6]:


labels = pd.read_csv("labels.csv")


# # Preprocessing

# In[7]:


labels = pd.read_csv("labels.csv")


# In[8]:


labels.dropna(inplace = True)
labels = labels.reset_index()


# In[9]:


for i in range(len(labels)):
    t = labels['text_corrected'][i]
    t = re.sub(r'[^\w\s]', '', t).lower()
    labels['text_corrected'][i] = t


# In[10]:


labels['overall_sentiment'] = labels['overall_sentiment'].replace({'very_positive':'positive', 'very_negative':'negative'})


# In[11]:


to_use_df = labels[['image_name','text_corrected', 'overall_sentiment']]


# In[12]:


le = LabelEncoder()
to_use_df['overall_sentiment'] = le.fit_transform(to_use_df['overall_sentiment'])
to_use_df.to_csv('text_df.csv', index=False)


# In[ ]:





# # Creating Data Loader

# In[13]:


mean = [0.4, 0.5, 0.5]
std = [0.4, 0.5, 0.5]


transform = transforms.Compose(
    [
     transforms.Resize((256, 256)),  
     transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])


# In[14]:


def pad_tensor(t):
    t = torch.tensor(t)
    padding = 187 - t.size()[0]
    t = torch.nn.functional.pad(t, (0, padding))
    return t


# In[15]:


#CREDIT GOES TO aladdinpersson https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/custom_dataset_txt/loader_customtext.py

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            float(self.stoi[token]) if token in self.stoi else float(self.stoi["<UNK>"])
            for token in tokenized_text
        ]


# In[16]:


#CREDIT GOES TO aladdinpersson https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/custom_dataset_txt/loader_customtext.py

class MemeAnalyzer(Dataset):
    def __init__(self,  root_dir, csv_file, transform = None, freq_threshold = 5):
        self.df = pd.read_csv(csv_file)
        self.img = self.df['image_name']
        self.captions = self.df['text_corrected']
        self.root_dir = root_dir
        self.transform = transform
        
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(list(self.captions))

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root_dir, self.df.iloc[index, 0])).convert("RGB")
        y_label = torch.tensor(int(self.df.iloc[index, 2]))
        caption = self.captions[index]
        if self.transform:
            image = self.transform(image)
            
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        padded_text_tensor = pad_tensor(torch.tensor((numericalized_caption)))
        return (image, padded_text_tensor, y_label)


# In[17]:


print (256 * 256 * 3)


# # Creating architecture for Neural Networks

# In[18]:


#FUNCTION TAKEN FROM LAB TASK
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.flat = nn.Flatten()
        self.inn = nn.Linear(196608, 256)
        self.hidden1=nn.Linear(256, 128)
        self.hidden2=nn.Linear(128, 64)
        self.outt = nn.Linear(64, 3)


    

    def forward(self, x):
      # Pass the input tensor through each of our operations
        #print(x.shape)
        x = self.flat(x)
        #print(x.shape)
        x = torch.sigmoid(self.inn(x))
        x = torch.sigmoid(self.hidden1(x))
        x = torch.sigmoid(self.hidden2(x))
        x = self.outt(x)
      
        return x 


# In[19]:


#FUNCTION TAKEN FROM LAB TASK
class NN_text(nn.Module):
    def __init__(self):
        super(NN_text, self).__init__()
        self.flat = nn.Flatten()
        self.inn = nn.Linear(187, 256)
        self.hidden1=nn.Linear(256, 128)
        self.hidden2=nn.Linear(128, 64)
        self.outt = nn.Linear(64, 3)


    

    def forward(self, x):
      # Pass the input tensor through each of our operations
        x = self.flat(x)
        x = torch.sigmoid(self.inn(x))
        x = torch.sigmoid(self.hidden1(x))
        x = torch.sigmoid(self.hidden2(x))
        x = self.outt(x)
      
        return x 


# In[20]:


#FUNCTION TAKEN FROM LAB TASK
class Combined_model(nn.Module):
    def __init__(self, modelA, modelB):
        super(Combined_model, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(6, 3)
        
    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x


# In[21]:


net = NN().to(device)
net_text = NN_text().to(device)


# In[22]:



combined_model = Combined_model(net, net_text).to(device)


# In[23]:


dataset = MemeAnalyzer(root_dir = './images', csv_file = 'text_data.csv', transform = transform)


# In[24]:


train_set, test_set = torch.utils.data.random_split(dataset, [5000, 1830])
train_loader = DataLoader(dataset = train_set, batch_size = 32, shuffle=True)
test_loader = DataLoader(dataset = test_set, batch_size = 32, shuffle=True)


# In[25]:


#FUNCTION TAKEN FROM LAB TASK

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, T, y) in enumerate(dataloader):
        X = X.to(device)
        T = T.to(device)
        y = y.to(device)
        pred = model(X,T)

        loss = loss_fn(pred, y)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            
#FUNCTION TAKEN FROM LAB TASK

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, T, y in dataloader:
            X = X.to(device)
            T = T.to(device)
            y = y.to(device)
            pred = model(X,T)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# In[26]:


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.001)

epochs = 2
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, combined_model, loss_fn, optimizer)
    test_loop(test_loader, combined_model, loss_fn)
print("Done!")

torch.save(combined_model.state_dict(), "combined_model_weights.pth")



# In[ ]:




