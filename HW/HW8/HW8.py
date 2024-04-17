# HW_08_ANN
# Nicholas DeRobertis
# I Pledge my Honor That I have Abided by the Stevens Honor System
# CWID: 20006069
'''
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# Load the dataset
df = pd.read_csv("C:/users/nickd/cs513/HW/HW_07_SVM/wisc_bc_ContinuousVar.csv")

# Convert Diagnosis Class column to categorical
df['diagnosis'] = df['diagnosis'].map({'B': 'Benign', 'M': 'Malignant'})

# Split the data set into training and testing
X = df.drop(['diagnosis', 'id'], axis=1)
Y = df['diagnosis']

# Normalize the Axes
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test Sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Define ANN and train the model with 5 nodes in the hidden layer
ANN = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000)
ANN.fit(X_train, Y_train)

# Creating and Printing Prediction
Y_pred = ANN.predict(X_test)
print("Prediction:", Y_pred)

# Calculate error rate
error_rate = np.mean(Y_pred != Y_test)
print('\n', "Error Rate:", error_rate)

# Make Confusion Matrix and Calculate Accuracy
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)
accuracy = accuracy_score(Y_test, Y_pred) * 100
print("\nAccuracy:", accuracy)

'''
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv("C:/users/nickd/cs513/HW/HW_07_SVM/wisc_bc_ContinuousVar.csv")

# Convert Diagnosis Class column to categorical
df['diagnosis'] = df['diagnosis'].map({'B': 'Benign', 'M': 'Malignant'})

# Create train and test data
train_data, test_data = train_test_split(data_scaled, test_size=0.3)

# Train the ANN model
X_train = train_data.drop(['id', 'diagnosis'], axis=1)
y_train = train_data['diagnosis']
ann_model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000)
scaler = MinMaxScaler()
ann_model.fit(X_train, y_train)

# Predict using the ANN model
X_test = test_data.drop(['id', 'diagnosis'], axis=1)
predicted_values = ann_model.predict(X_test)
final_predictions = pd.Series(predicted_values).map({'B': 'Benign', 'M': 'Malignant'})

# Outputs confusion matrix and accuracy
ann_comptable = confusion_matrix(final_predictions, test_data['diagnosis'])
ann_acc = accuracy_score(test_data['diagnosis'], final_predictions)
print("Confusion Matrix:")
print(ann_comptable)
print("\nAccuracy:", ann_acc)