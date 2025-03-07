from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import fasttext
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from huggingface_hub import hf_hub_download
from sklearn.utils import resample


########################
### TRAIN SET IMPORT ###
########################
df = pd.read_csv('data/train_submission.csv')

# NaN deletion
df = df.dropna()

# Outliers deletion
df = df[df['Label'].map(df['Label'].value_counts()) > 1]

################################
### TRAIN & VALIDATION SPLIT ###
################################
X = df['Text']
y = df['Label']

# Train & validation split with same distribution
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


##################
### RESAMPLING ###
##################
# Combine X_train and y_train into a single DataFrame
train_df = pd.DataFrame({'Text': X_train, 'Label': y_train})

# Separate majority and minority classes
majority_class = train_df['Label'].value_counts().idxmax()
minority_classes = train_df['Label'].value_counts().index[train_df['Label'].value_counts() < train_df['Label'].value_counts().max()]

# Oversample minority classes
oversampled_dfs = []
for label in minority_classes:
    minority_df = train_df[train_df['Label'] == label]
    oversampled_df = resample(minority_df, replace=True, n_samples=train_df['Label'].value_counts().max(), random_state=42)
    oversampled_dfs.append(oversampled_df)

# Combine oversampled minority classes with the majority class
balanced_df = pd.concat([train_df[train_df['Label'] == majority_class]] + oversampled_dfs)

# Shuffle the balanced dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split back into X_train and y_train
X_train_balanced = balanced_df['Text']
y_train_balanced = balanced_df['Label']


#################################
### PREPARE DATA FOR FASTTEXT ###
#################################
def prepare_fasttext_data(texts, labels, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for text, label in zip(texts, labels):
            f.write(f"{text} __label__{label}\n")

# Save the training and validation data in FastText format
prepare_fasttext_data(X_train, y_train, 'models/FastText/fasttext_data/fasttext_train.txt')
prepare_fasttext_data(X_val, y_val, 'models/FastText/fasttext_data/fasttext_val.txt')
prepare_fasttext_data(X_train_balanced, y_train_balanced, 'models/FastText/fasttext_data/fasttext_train_balanced.txt')


################
### TRAINING ###
################
model = fasttext.train_supervised(
    input='models/FastText/fasttext_data/fasttext_train_balanced.txt',
    # input='models/FastText/fasttext_data/fasttext_train_balanced.txt',
    # epoch=30,               # Increase number of epochs
    # lr=0.05,               # Lower learning rate
    # wordNgrams=2,           # Use bigrams
    # dim=200,                # Larger word vectors
    # ws=8,                   # Larger context window
    # bucket=1000000,         # Reduce number of subword buckets
    autotuneValidationFile='models/FastText/fasttext_data/fasttext_val.txt',
    autotuneDuration=3600*3,
    # # Training arguments to handle class imbalance
    # minCountLabel=1,        # Ignore rare labels
    # loss='ova',             # Use one-vs-all loss for multi-label classification
    # neg=10,                 # Number of negative samples
)



##################
### EVALUATION ###
##################
validation_file = 'models/FastText/fasttext_data/fasttext_val.txt'
result = model.test(validation_file)
print(f"Accuracy of the fine-tuned: {result[1]}")


####################################
### TEST SET LABEL WITH FASTTEXT ###
####################################
# Generate predictions from the test_without_labels.csv file
df_test = pd.read_csv('data/test_without_labels.csv')
X_test = df_test['Text']

prepare_fasttext_data(X_test, [0]*len(X_test), 'models/FastText/fasttext_data/fasttext_test.txt')

predictions = []
with open('models/FastText/fasttext_data/fasttext_test.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if len(predictions) % 100 == 0:
            print(round(len(predictions)/len(X_test), 3), "%", " done", end='\r')

        # remove __label__0 and \n
        line = line.split('__label__')[0]
        predictions.append(model.predict(line))

df_test['Label'] = [label[0][0].split("__label__")[1] for label in predictions]

# Remove columns Usage and Text
df_test = df_test.drop(columns=['Usage', 'Text'])

# Add a column Id
df_test['ID'] = range(1, len(df_test) + 1)

# Reorder columns
df_test = df_test[['ID', 'Label']]
df_test.to_csv('results/fasttext_predictions.csv', index=False)





##############################
### BENCHMARK WITH GLOTLID ###
##############################
# model.bin is the latest version always
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
model = fasttext.load_model(model_path)

# Evaluate the model manually on the validation df
y_pred = X_val.apply(lambda x: model.predict(x)[0][0])

label_pred = y_pred.apply(lambda x: x.split('__label__')[1].split('_')[0])

accuracy = (label_pred == y_val).mean()
print(f"Accuracy of the zero-shot model: {accuracy}")