# Midterm Question 5
# Nicholas DeRobertis
# I Pledge my Honor That I have Abided by the Stevens Honor System
# CWID: 20006069

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv(r"C:/users/nickd/CS513/Midterm/Admission_v2.csv")

# Replaces any blank spaces with NaN and removes any rows without values
df.replace(' ', np.nan, inplace=True)
df.dropna(inplace=True)

# Create X and Y Axes
X = df.drop(['GRE', 'GPA','ADMIT'], axis=1)
Y = df['ADMIT']

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

# Make Confusion Matrix and Calculate Accuracy For K=3
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("\n")
print("Confusion Matrix k=3:")
print(conf_matrix)
accuracy = accuracy_score(Y_test, Y_pred) * 100
print ("Accuracy k=3:", accuracy)

# K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

# Make Confusion Matrix and Calculate Accuracy For K=5
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("\n")
print("Confusion Matrix k=5:")
print(conf_matrix)
accuracy = accuracy_score(Y_test, Y_pred) * 100
print ("Accuracy k=5:", accuracy)

# K=7
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

# Make Confusion Matrix and Calculate Accuracy For K=7
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("\n")
print("Confusion Matrix k=7:")
print(conf_matrix)
accuracy = accuracy_score(Y_test, Y_pred) * 100
print ("Accuracy k=7:", accuracy)



