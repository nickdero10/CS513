# HW_05_CART
# Nicholas DeRobertis
# I Pledge my Honor That I have Abided by the Stevens Honor System
# CWID: 20006069

import pandas as pd
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree
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

# Implementing CART
cart = tree.DecisionTreeClassifier()

# Train the model
cart.fit(X_train, Y_train)

# Make Tree Structure
tr = export_graphviz(cart, out_file = None, feature_names=X.columns, class_names=Y.unique())
graph = graphviz.Source(tr).view()

# Predicting class for test set
Y_pred = cart.predict(X_test)
print('\n', "Prediction: ", Y_pred)

# Make Confusion Matrix and Calculate Accuracy
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)
accuracy = accuracy_score(Y_test, Y_pred) * 100
print("\nAccuracy:", accuracy)

# Resources Used
# https://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart
# https://www.tutorialspoint.com/scikit_learn/scikit_learn_decision_trees.htm 
# https://www.youtube.com/watch?v=ZbYuL6tHV1g 
# ChatGPT
