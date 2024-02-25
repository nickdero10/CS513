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

