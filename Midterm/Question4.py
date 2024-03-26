# Nicholas DeRobertis
# Midterm Question 4

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

# Define the bins for discretization
bins = [0, 500, 600, 700, float('inf')]  
labels = ["Up to 500 inclusive", "Above 500 and up to 600", 
            "Above 600 and up to 700", "Above 700"]

# Discretize the 'GRE' scores using pd.cut()
df['GRE Category'] = pd.cut(df['GRE'], bins=bins, labels=labels, right=False)

gpa_bins = [0, 2.5, 3, 3.5, float('inf')]
gpa_labels = ["Up to 2.5 inclusive", "Above 2.5 and up to 3",
                "Above 3 and up to 3.5", "Over 3.5"]

# Discretize the 'GPA' scores using pd.cut()
df['GPA Category'] = pd.cut(df['GPA'], bins=gpa_bins, labels=gpa_labels, right=False)

# Display the first few rows of the dataframe to verify the changes
print(df)

# Count the number of entries in each GPA category
gre_counts = df['GRE Category'].value_counts()

# Display the counts
print("Counts of entries in each GRE category:")
print(gre_counts, '\n')

# Count the number of entries in each GPA category
gpa_counts = df['GPA Category'].value_counts()

# Display the counts
print("Counts of entries in each GPA category:")
print(gpa_counts)