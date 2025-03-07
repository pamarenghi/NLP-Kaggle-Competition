import pandas as pd
from logistic_regression import model, vectorizer

#######################
### TEST SET IMPORT ###
#######################
data_path = "data/test_without_labels.csv"
df = pd.read_csv(data_path)

# Text column
if "Text" in df.columns:
    X_test = df["Text"]
else:
    raise ValueError("La colonne 'Text' n'est pas présente dans le CSV.")

################
### ENCODING ###
################
X_test_tfidf = vectorizer.transform(X_test)



###################
### PREDICTIONS ###
###################
predictions = model.predict(X_test_tfidf)


###################
### OUTPUT FILE ###
###################
output_df = pd.DataFrame({
    "ID": df.index + 1,
    "Label": predictions
})

output_df.to_csv("results/logistic_predictions.csv", index=False)
print("Les prédictions ont été enregistrées dans results/logistic_predictions.csv")
