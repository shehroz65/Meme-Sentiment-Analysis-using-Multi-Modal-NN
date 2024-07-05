import pandas as pd
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from PIL import Image
import spacy
import numpy as np
import re
import os
import torch.nn.functional as F
import torch.optim as optim
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader

# Load SpaCy model and prefer GPU if available
spacy.prefer_gpu()
spacy_eng = spacy.load("en_core_web_sm")

# Define device for training; will use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read and preprocess labels DataFrame
labels = pd.read_csv("labels.csv")
labels.dropna(inplace=True)
labels['text_corrected'] = labels['text_corrected'].apply(lambda x: re.sub(r'[^\w\s]', '', x).lower())
labels['overall_sentiment'] = labels['overall_sentiment'].replace({'very_positive':'positive', 'very_negative':'negative'})

# Encode sentiment labels
le = LabelEncoder()
labels['overall_sentiment'] = le.fit_transform(labels['overall_sentiment'])

# Save cleaned DataFrame for training
labels[['image_name', 'text_corrected', 'overall_sentiment']].to_csv('text_df.csv', index=False)

# Image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.4, 0.5, 0.5], [0.4, 0.5, 0.5])
])

# Vocabulary and Dataset classes
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def tokenizer_eng(self, text):
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
        return [self.stoi.get(token, self.stoi['<UNK>']) for token in self.tokenizer_eng(text)]

class MemeAnalyzer(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, freq_threshold=5):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.df['text_corrected'].tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.df.loc[index, 'image_name'])
        image = Image.open(img_path).convert('RGBA').convert('RGB')
        if self.transform:
            image = self.transform(image)
        caption = self.df.loc[index, 'text_corrected']
        numericalized_caption = [self.vocab.stoi['<SOS>']] + self.vocab.numericalize(caption) + [self.vocab.stoi['<EOS>']]
        padded_text_tensor = F.pad(torch.tensor(numericalized_caption, dtype=torch.float32), (0, 187 - len(numericalized_caption)))  # Ensure this tensor is float
        return image.to(device), padded_text_tensor.to(device), torch.tensor(int(self.df.loc[index, 'overall_sentiment']), dtype=torch.float32).to(device)  # Ensure this tensor is float


# Neural Network Definitions
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(196608, 256)  # Adjust the input size according to your flattened image tensor size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 3)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.out(x)
        return x

class NN_text(nn.Module):
    def __init__(self):
        super(NN_text, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(187, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 3)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.out(x)
        return x

class CombinedModel(nn.Module):
    def __init__(self, modelA, modelB):
        super(CombinedModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(6, 3)

    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x

# Load the dataset
dataset = MemeAnalyzer(root_dir='./images', csv_file='text_df.csv', transform=transform)

# Split the dataset
train_set, test_set = torch.utils.data.random_split(dataset, [5000, 1830])

# DataLoader setup
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

# Instantiate models and move them to GPU
net = NN().to(device)
net_text = NN_text().to(device)
combined_model = CombinedModel(net, net_text).to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(combined_model.parameters(), lr=0.001)

# Training and testing loops
def train_loop(dataloader, model, loss_fn, optimizer):
    for batch, (X, T, y) in enumerate(dataloader):
        X = X.to(device).float()  # Ensure input is float
        T = T.to(device).float()  # Ensure input is float
        y = y.to(device).long()   # Ensure labels are long for CrossEntropyLoss
        pred = model(X, T)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{batch * len(X):>5d}/{len(dataloader.dataset):>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, T, y in dataloader:
            X = X.to(device).float()  # Ensure X is float
            T = T.to(device).float()  # Ensure T is float
            y = y.to(device).long()   # Ensure y is long for CrossEntropyLoss
            pred = model(X, T)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    print(f"Pred shape: {pred.shape}, dtype: {pred.dtype}")
    print(f"Labels shape: {y.shape}, dtype: {y.dtype}")

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")


# Run training and testing
epochs = 2
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, combined_model, loss_fn, optimizer)
    test_loop(test_loader, combined_model, loss_fn)
print("Done!")

# Save the model
torch.save(combined_model.state_dict(), "combined_model.pth")
