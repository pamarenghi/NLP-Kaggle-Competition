import pandas as pd
from naive_bayes import model, label_encoder


##############
### IMPORT ###
##############
data_path = "data/test_without_labels.csv"
df = pd.read_csv(data_path) 

# Text column
if "Text" in df.columns:
    X_test = df["Text"]
else:
    raise ValueError("La colonne 'Text' n'est pas présente dans le CSV.")



##################
### PREDICTION ###
##################
print("Making predictions...")
predictions = model.predict(X_test) 
predictions = label_encoder.inverse_transform(predictions)
print("Predictions finished successfully!\nSaving predictions...")



###################
### OUTPUT FILE ###
###################
output_df = pd.DataFrame({
    "ID": df.index + 1,
    "Label": predictions
})

# Save file 
output_df.to_csv("results/bayes_predictions.csv", index=False)
print("Les prédictions ont été enregistrées dans results")
