# HW_07_SVM
# Nicholas DeRobertis
# I Pledge my Honor That I have Abided by the Stevens Honor System
# CWID: 20006069

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# Load data
df = pd.read_csv("C:/users/nickd/cs513/HW/HW_07_SVM/wisc_bc_ContinuousVar.csv")


# Convert Diagnosis Class column to categorical
df['diagnosis'] = df['diagnosis'].map({'B': 'Benign', 'M': 'Malignant'})

# Generate train and test sets
X = df.drop(columns=['diagnosis'])
Y = df['diagnosis']

# Normalize the Axes
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into Train and Test Sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

# Define SVM and Train the model
SVM = SVC()
SVM.fit(X_train, Y_train)

# Creating and Printing Prediction
Y_pred = SVM.predict(X_test)
print('\n', "Prediction:", Y_pred)


# Make Confusion Matrix and Calculate Accuracy
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)
accuracy = accuracy_score(Y_test, Y_pred) * 100
print("\nAccuracy:", accuracy)

# Resources:
# https://scikit-learn.org/stable/modules/svm.html
# ChatGPT