import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle
import os

from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


##############
### IMPORT ###
##############
train_path = "data/train_submission.csv"
df_train_set = pd.read_csv(train_path)

# Remove NaN values
df_train_set = df_train_set.dropna()

# Drop duplicate (Label, Text) tuples
df_train_set = df_train_set.drop_duplicates(['Label', 'Text'])

# Remove labels with only one occurrence
label_counts = df_train_set['Label'].value_counts()
labels_to_remove = label_counts[label_counts < 2].index
df_train_set = df_train_set[~df_train_set['Label'].isin(labels_to_remove)]

# Lowercase
df_train_set["Text"] = df_train_set["Text"].apply(func=lambda x:x.lower())

# Train-test split
df_train, df_validation = train_test_split(df_train_set, test_size=0.2, stratify=df_train_set["Label"], random_state=42)


################
### ENCODING ###
################
label_encoder = LabelEncoder()
df_train['category_encoded'] = label_encoder.fit_transform(df_train['Label'])
df_validation['category_encoded'] = label_encoder.transform(df_validation['Label'])
num_classes = len(label_encoder.classes_)

# Features and labels
X_train, y_train = df_train["Text"], df_train['category_encoded']
X_valid, y_valid = df_validation["Text"], df_validation['category_encoded'] 

class_weights = compute_sample_weight("balanced", y_train)

print("Preprocessing done")


################
### TRAINING ###
################
model_filename = 'models/naive_bayes/naive_bayes_model.pkl'
if os.path.exists(model_filename):
    print("Found model\nLoading model...")
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully!")
else:
    print("No model foun\nCreating model...")
    # Training logistic regression
    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5)), MultinomialNB()) #to capture ponctuation
    model.fit(X_train, y_train, multinomialnb__sample_weight=class_weights)
    
    print("Model created!\nSaving model...")
    # Download
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print("Model saved successfully!")


##################
### EVALUATION ###
##################
def eval_model(y_pred):
    accuracy = accuracy_score(y_valid, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_valid, y_pred, average='weighted', zero_division=0)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

# Predictions
print("Evaluating model...")
y_pred = model.predict(X_valid)
eval_model(y_pred)
# Accuracy: 0.7930
# Precision: 0.8361
# Recall: 0.7930
# F1 Score: 0.7858