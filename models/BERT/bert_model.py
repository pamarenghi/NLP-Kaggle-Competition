import pandas as pd
import numpy as np
import pickle
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



########################
### HYPER PARAMETERS ###
########################
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MODEL_NAME = "distilbert-base-multilingual-cased"

device = torch.device("mps" if torch.backends.mps.is_available() else 
                      ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Utilisation de: {device}")


#################
### ENCODDING ###
#################
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


############
### BERT ###
############
class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Utilisation du token [CLS]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits



################
### TRAINING ###
################
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)



##################
### EVALUATION ###
##################
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    return accuracy_score(actual_labels, predictions)



#######################
### ENTIRE PIPELINE ###
#######################
# Data import
print("Chargement des données...")
df = pd.read_csv('data/train_submission.csv')
df = df.dropna()
df = df[df['Label'].map(df['Label'].value_counts()) > 1]

# Label encoding
label_dict = {label: idx for idx, label in enumerate(df['Label'].unique())}
df['label_id'] = df['Label'].map(label_dict)

with open('models/BERT/label_mapping.pkl', 'wb') as f:
    pickle.dump(label_dict, f)

# Train & validation split
X = df['Text']
y = df['label_id']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=df['Label'], random_state=42)

# Model
print("Chargement du modèle BERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME)
model = BertClassifier(base_model, len(label_dict))
model.to(device)

# Feature encoding
train_dataset = TextDataset(X_train, y_train, tokenizer, MAX_LENGTH)
val_dataset = TextDataset(X_val, y_val, tokenizer, MAX_LENGTH)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
model_file = 'models/BERT/bert_model.pt'
if os.path.exists(model_file):
    print(f"Chargement du modèle existant: {model_file}")
    model.load_state_dict(torch.load(model_file, map_location=device))
else:
    print("Début de l'entraînement...")
    best_accuracy = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        val_accuracy = evaluate(model, val_dataloader, device)
        
        print(f"Loss: {train_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), model_file)
            print(f"Modèle sauvegardé avec accuracy: {best_accuracy:.4f}")

# Evaluation
print("\nÉvaluation finale du modèle...")
val_accuracy = evaluate(model, val_dataloader, device)
print(f"Accuracy: {val_accuracy:.4f}")

###########################
### INFERENCE FUNCTIONS ###
###########################
def predict(text, model, tokenizer, label_dict):
    id_to_label = {v: k for k, v in label_dict.items()}
    
    # Tokenisation
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    # Prediction
    model.eval()
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    
    return id_to_label[preds.item()]

print("\nLe modèle est prêt à être utilisé pour les prédictions.")