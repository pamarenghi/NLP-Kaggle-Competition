import pandas as pd
import torch
import pickle
import os
from transformers import AutoTokenizer, AutoModel
from torch import nn

# Config
MAX_LENGTH = 128
MODEL_NAME = "distilbert-base-multilingual-cased"

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Utilisation de: {device}")

# BERT
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



######################
### LABEL TEST SET ###
######################
# Mapping loading
if not os.path.exists('label_mapping.pkl'):
    raise FileNotFoundError("Fichier label_mapping.pkl non trouvé. Assurez-vous d'avoir entraîné le modèle.")

with open('label_mapping.pkl', 'rb') as f:
    label_dict = pickle.load(f)

id_to_label = {v: k for k, v in label_dict.items()}

# Tokeniser and model loading
print("Chargement du tokenizer et du modèle BERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME)
model = BertClassifier(base_model, len(label_dict))

model_file = 'bert_model.pt'
if not os.path.exists(model_file):
    raise FileNotFoundError(f"Fichier modèle {model_file} non trouvé. Assurez-vous d'avoir entraîné le modèle.")

model.load_state_dict(torch.load(model_file, map_location=device))
model.to(device)
model.eval()

# Data import
data_path = "data/test_without_labels.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Fichier {data_path} non trouvé.")

df = pd.read_csv(data_path)

# Label selection
if "Text" in df.columns:
    texts = df["Text"]
else:
    raise ValueError("La colonne 'Text' n'est pas présente dans le CSV.")

# Predictions
print(f"Génération des prédictions pour {len(texts)} textes...")
predictions = []
batch_size = 32
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size].tolist()
    
    # Tokenizer
    encodings = tokenizer(
        batch_texts,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    # Prediction
    with torch.no_grad():
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

        batch_predictions = [id_to_label[pred_id.item()] for pred_id in preds]
        predictions.extend(batch_predictions)
    
    if (i + batch_size) % (5 * batch_size) == 0 or (i + batch_size) >= len(texts):
        print(f"  Traité {min(i + batch_size, len(texts))}/{len(texts)} textes")

# Create output df
output_df = pd.DataFrame({
    "ID": df.index + 1,
    "Label": predictions
})

# Save result file
output_file = "predictions_bert.csv"
output_df.to_csv(output_file, index=False)
print(f"Les prédictions ont été enregistrées dans {output_file}")