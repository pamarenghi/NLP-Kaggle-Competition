import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Fetch train csv
df = pd.read_csv('data/train_submission.csv')

# NaN deletion
df = df.dropna()

# Outliers deletion
df = df[df['Label'].map(df['Label'].value_counts()) > 1]

# X, y separation
X = df['Text']
y = df['Label']

# Train and validation separation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Vectorizing with TFIDF
vectorizer = TfidfVectorizer(max_features=90000, min_df=2, max_df=0.95, dtype=np.float32)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Download model parameters
model_filename = 'models/logistic_regression/logistic_model.pkl'
if os.path.exists(model_filename):
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
else:
    # Training logistic regression
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    
    # Download
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

# Prediction and evaluation
y_pred = model.predict(X_val_tfidf)
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy:.4f}')
