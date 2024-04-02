# HW_06_RF
# Nicholas DeRobertis
# I Pledge my Honor That I have Abided by the Stevens Honor System
# CWID: 20006069

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
df = pd.read_csv("C:/users/nickd/cs513/HW/HW_02_EDA/breast-cancer-wisconsin.csv")

# Set column names
df.columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
              'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
              'Normal Nucleoli', 'Mitoses', 'Diagnosis Class']

# Replaces any ? with NaN and removes any rows without values
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Make Bare Nucleii Numeric
df['Bare Nuclei'] = pd.to_numeric(df['Bare Nuclei'])

# Convert Diagnosis Class column to categorical
df['Diagnosis Class'] = df['Diagnosis Class'].replace({2: 'Benign', 4: 'Malignant'})

# Generate train and test sets
X = df.drop(columns=['Sample code number', 'Diagnosis Class'])
Y = df['Diagnosis Class']

# Normalize the Axes
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into Train and Test Sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

# Fit Random Forest model
fit = RandomForestClassifier(n_estimators=1000, random_state=42)
fit.fit(X_train, Y_train)

# Feature importance
importance = fit.feature_importances_
print("Feature Importance:")
for feature, importance_score in zip(X.columns, importance):
    print(f"{feature}: {importance_score}")

# Creating and Printing Prediction
prediction = fit.predict(X_test)
print('\n', "Prediction:", prediction)

# Creating and Printing Accuracy
accuracy = accuracy_score(Y_test, prediction)
print('\n', "Accuracy:", accuracy)

# Creating and Printing Error rate
errorRate = 1 - accuracy
print('\n',"Error Rate:", errorRate)

# Recourses:
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# ChatGPT