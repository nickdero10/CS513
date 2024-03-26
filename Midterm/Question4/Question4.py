# Midterm Question 4
# Nicholas DeRobertis
# I Pledge my Honor That I have Abided by the Stevens Honor System
# CWID: 20006069

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv(r"C:/users/nickd/CS513/Midterm/Admission_v2.csv")

# Replaces any blank spaces with NaN and removes any rows without values
df.replace(' ', np.nan, inplace=True)
df.dropna(inplace=True)

# Define the mapping dictionary for GRE
GRE_map = {
    'Up to 500 Inclusive': lambda x: x <= 500,
    'Above 500 and up to 600': lambda x: 500 < x <= 600,
    'Above 600 and up to 700': lambda x: 600 < x <= 700,
    'Above 700': lambda x: x > 700
}

# Apply the mapping for GRE using map function
df['GRE'] = df['GRE'].map(lambda x: next((key for key, condition in GRE_map.items() if condition(x)), 'Unknown'))

# Define the mapping dictionary for GPA
GPA_map = {
    'Up to 2.5 Inclusive': lambda x: x <= 2.5,
    'Above 2.5 and up to 3': lambda x: 2.5 < x <= 3,
    'Above 3 and up to 3.5': lambda x: 3 < x <= 3.5,
    'Above 3.5': lambda x: x > 3.5
}

# Apply the mapping for GPA using map function
df['GPA'] = df['GPA'].map(lambda x: next((key for key, condition in GPA_map.items() if condition(x)), 'Unknown'))
print(df)

# Create X and Y Axes
X = df.drop(['GRE', 'GPA','ADMIT'], axis=1)
Y = df['ADMIT']

# Normalize the Axes
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into Train and Test Sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, 
                                                    test_size=0.3, random_state=42)

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
print("\nAccuracy:", accuracy, '\n')