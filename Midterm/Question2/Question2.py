# Midterm Question 2
# Nicholas DeRobertis
# I Pledge my Honor That I have Abided by the Stevens Honor System
# CWID: 20006069

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:/users/nickd/CS513/Midterm/Admission_v2_missing.csv")

# Summarizes the columns
print("\n")
print("Admission Data Summarized")
print(df.describe())

# Replaces any blank space with NA and counts how many NA are in the data
df.replace(' ', pd.NA, inplace=True)
print("\n")
print("Number of Blanks in the Data")
print(df.isna().sum())

# Convert GRE Column to numeric
df['GRE'] = pd.to_numeric(df['GRE'], errors = 'coerce')

# Convert GPA Column to numeric
df['GPA'] = pd.to_numeric(df['GPA'], errors = 'coerce')

# Fills in missing values with the mean of the column
numeric_columns = df.select_dtypes(include='number').columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
df = df.round(2)

# Scatter Plot GRE vs. GPA
pd.plotting.scatter_matrix(df.iloc[:, 2:4], figsize=(800,4))
plt.suptitle("Scatter Plot of GRE vs. GPA")
plt.show()

# Scatter Plot GRE vs. Rank
pd.plotting.scatter_matrix(df.iloc[:, [2,4]], figsize=(800,4))
plt.suptitle("Scatter Plot of GRE vs. Rank")
plt.show()

# Scatter Plot GPA vs. Rank
pd.plotting.scatter_matrix(df.iloc[:, 3:5], figsize=(4,4))
plt.suptitle("Scatter Plot of GPA vs. Rank")
plt.show()

# Box Plot GRE
df.iloc[:, 2].plot(kind='box')
plt.title("Box Plot of GRE")
plt.show()

# Box Plot GPA
df.iloc[:, 3].plot(kind='box')
plt.title("Box Plot of GPA")
plt.show()



