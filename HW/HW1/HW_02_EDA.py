#HW_02_EDA
#Nicholas DeRobertis
#I Pledge my Honor That I have Abided by the Stevens Honor System
#CWID: 20006069



import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:/users/nickd/cs513/HW/HW1/breast-cancer-wisconsin.csv")

#Summarizes the columns
print("\n")
print("Breast Cancer Wisconsin Data Summarized")
print("\n")
print(df.describe())

#Replaces any ? with NA and counts how many NA are in the data
df.replace('?', pd.NA, inplace=True)
print("\n")
print("Number of Appearances of ? in the Data")
print(df.isna().sum())

numeric_columns = df.select_dtypes(include='number').columns

df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

df = df.round(2)

#Class Vs. F6 Table
table = pd.crosstab(df['Class'], df['F6'])
print("\n")
print("Frequency Table of Class Vs. F6")
print(table)

#Scatter Plot F1 to F6
pd.plotting.scatter_matrix(df.iloc[:, 1:7], figsize=(10,10))
plt.suptitle("Scatter Plot of F1 to F6")
plt.show()

#Histogram F7 to F9
df.iloc[:, 7:10].plot(kind='box')
plt.title("Histogram Box Plot of F7 to F9")
plt.show()

#Reloads the data
df = pd.read_csv(r"C:/users/nickd/cs513/HW/HW1/breast-cancer-wisconsin.csv")

#Removes any rows without values
df.dropna(inplace=True)

print("\n")
print("Reloaded Data with Rows Wihtout Data Removed")
print(df)

#Resources Used
#Excel to Python https://datatofish.com/read_excel/
#Chat-GPT
