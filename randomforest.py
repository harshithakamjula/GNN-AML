import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

from google.colab import drive

drive.mount('/content/drive')
df = pl.read_csv("Formattedtransactions/Small_HI.csv")

# Importing Scikit-Learn's train_test_split function
y_polars = df['Is Laundering']
X_polars = df.drop(columns = ['Is Laundering'])
from sklearn.model_selection import train_test_split

# Performing a train-validation split on the Polars data
X_train_polars, X_val_polars, y_train_polars, y_val_polars = train_test_split(X_polars, y_polars, test_size = 0.2, random_state = 42)

from sklearn.ensemble import RandomForestClassifier
# Instantiating a Random Forest Classifier object for the Polars DataFrame
rfc_model_polars = RandomForestClassifier(n_estimators = 50,
                                          max_depth = 20,
                                          min_samples_split = 10,
                                          min_samples_leaf = 2)

# Fitting the Polars DataFrame to the Random Forest Classifier algorithm
rfc_model_polars.fit(X_train_polars, y_train_polars)

y_pred = rfc_model_polars.predict(X_val_polars)

from sklearn.metrics import accuracy_score, f1_score
accuracy = accuracy_score(y_val_polars, y_pred)

f1 = f1_score(y_val_polars, y_pred)
print(f1)

from joblib import dump, load
dump(rfc_model_polars, 'rfc_model_polars.joblib')

df = pl.read_csv("Formattedtransactions/Small_LI.csv")

# Importing Scikit-Learn's train_test_split function
y_polars = df['Is Laundering']
X_polars = df.drop(columns = ['Is Laundering'])
from sklearn.model_selection import train_test_split

# Performing a train-validation split on the Polars data
X_train_polars, X_val_polars, y_train_polars, y_val_polars = train_test_split(X_polars, y_polars, test_size = 0.2, random_state = 42)

loaded_model = load('rfc_model_polars.joblib')

# Step 4: Retrain the loaded model with the new dataset
loaded_model.fit(X_train_polars, y_train_polars)

y_pred = loaded_model.predict(X_val_polars)

accuracy = accuracy_score(y_val_polars, y_pred)
f1 = f1_score(y_val_polars, y_pred)
print(f1)

dump(rfc_model_polars, 'rfc_model_polars.joblib')

df = pl.read_csv("Formattedtransactions/Medium_HI.csv")

# Importing Scikit-Learn's train_test_split function
y_polars = df['Is Laundering']
X_polars = df.drop(columns = ['Is Laundering'])
from sklearn.model_selection import train_test_split

# Performing a train-validation split on the Polars data
X_train_polars, X_val_polars, y_train_polars, y_val_polars = train_test_split(X_polars, y_polars, test_size = 0.2, random_state = 42)

loaded_model = load('rfc_model_polars.joblib')

# Step 4: Retrain the loaded model with the new dataset
loaded_model.fit(X_train_polars, y_train_polars)

y_pred = loaded_model.predict(X_val_polars)
f1 = f1_score(y_val_polars, y_pred)
print(f1)

dump(rfc_model_polars, 'rfc_model_polars.joblib')

df = pl.read_csv("Formattedtransactions/Medium_LI.csv")

# Importing Scikit-Learn's train_test_split function
y_polars = df['Is Laundering']
X_polars = df.drop(columns = ['Is Laundering'])
from sklearn.model_selection import train_test_split

# Performing a train-validation split on the Polars data
X_train_polars, X_val_polars, y_train_polars, y_val_polars = train_test_split(X_polars, y_polars, test_size = 0.2, random_state = 42)

loaded_model = load('rfc_model_polars.joblib')

# Step 4: Retrain the loaded model with the new dataset
loaded_model.fit(X_train_polars, y_train_polars)

y_pred = loaded_model.predict(X_val_polars)

f1 = f1_score(y_val_polars, y_pred)
print(f1)

