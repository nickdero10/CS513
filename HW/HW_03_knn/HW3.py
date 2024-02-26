# HW_03_knn
# Nicholas DeRobertis
# I Pledge my Honor That I have Abided by the Stevens Honor System
# CWID: 20006069

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv(r"C:/users/nickd/cs513/HW/HW_02_EDA/breast-cancer-wisconsin.csv")

# Name the columns
df.columns = ['Sample Code Number', 'Clump Thickness', 'Uniformity of Cell Size',
              'Uniformity of Cell Shape', 'Marginal Adhesion', 
              'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
              'Normal Nuclei', 'Mitoses', 'Diagnosis Class']

# Replaces any ? with NaN and removes any rows without values
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Make Bare Nucleii Numeric
df['Bare Nuclei'] = pd.to_numeric(df['Bare Nuclei'])

# Make Class Binary Values
df['Diagnosis Class'] = df['Diagnosis Class'].map({2: 'Benign', 4: 'Maligant'})

# Create X and Y Axes
X = df.drop(['Sample Code Number', 'Diagnosis Class'], axis=1)
Y = df['Diagnosis Class']

# Normalize the Axes
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into Train and Test Sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, 
                                                    test_size=0.3, random_state=42)

# Define KNearest-Neighbor Classifier
knn = KNeighborsClassifier()

# K=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

# Make Condusion Matrix and Calculate Accuracy For K=3
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("\n")
print("Confusion Matrix k=3:")
print(conf_matrix)
print("\n")
accuracy = accuracy_score(Y_test, Y_pred) * 100
print ("Accuracy k=3:", accuracy)

#K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

# Make Condusion Matrix and Calculate Accuracy For K=5
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("\n")
print("Confusion Matrix k=5:")
print(conf_matrix)
print("\n")
accuracy = accuracy_score(Y_test, Y_pred) * 100
print ("Accuracy k=5:", accuracy)

#K=10
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

# Make Condusion Matrix and Calculate Accuracy For K=10
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("\n")
print("Confusion Matrix k=10:")
print(conf_matrix)
print("\n")
accuracy = accuracy_score(Y_test, Y_pred) * 100
print ("Accuracy k=10:", accuracy)
