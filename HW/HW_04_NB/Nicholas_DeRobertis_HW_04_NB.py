# HW_04_NB
# Nicholas DeRobertis
# I Pledge my Honor That I have Abided by the Stevens Honor System
# CWID: 20006069

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv("C:/users/nickd/cs513/HW/HW_02_EDA/breast-cancer-wisconsin.csv")

# Name the columns
df.columns = ['Sample Code Number', 'Clump Thickness', 'Uniformity of Cell Size',
              'Uniformity of Cell Shape', 'Marginal Adhesion', 
              'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
              'Normal Nucleoli', 'Mitoses', 'Diagnosis Class']

# Replaces any ? with NaN and removes any rows without values
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Make Bare Nucleii Numeric
df['Bare Nuclei'] = pd.to_numeric(df['Bare Nuclei'])

# Make Class Binary Values
df['Diagnosis Class'] = df['Diagnosis Class'].map({2: 'Benign', 4: 'Malignant'})

# Create X and Y Axes
X = df.drop(['Sample Code Number', 'Diagnosis Class'], axis=1)
Y = df['Diagnosis Class']

# Normalize the Axes
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into Train and Test Sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

# Define Naïve Bayes Classifier
nb = GaussianNB()

# Train the model
nb.fit(X_train, Y_train)

# Predict the classes
Y_pred = nb.predict(X_test)

# Make Confusion Matrix and Calculate Accuracy
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)
accuracy = accuracy_score(Y_test, Y_pred) * 100
print("\nAccuracy:", accuracy)