# HW_06_C5.0
# Nicholas DeRobertis
# I Pledge my Honor That I have Abided by the Stevens Honor System
# CWID: 20006069

rm(list=ls())

library(C50)

# Load data, Remove blank values
Breast_Cancer<- read.csv("C:/users/nickd/cs513/HW/HW_02_EDA/breast-cancer-wisconsin.csv", na.strings="?")

# Set value 2 to benign and 4 to Malignant
Breast_Cancer$Class <- factor(Breast_Cancer$Class, levels = c(2,4),labels = c("Benign", "Malignant"))
is.factor(Breast_Cancer$Class)

# Loading 70% Breast cancer record in training dataset
idx<-sort(sample(nrow(Breast_Cancer),as.integer(.70*nrow(Breast_Cancer))))
training<-Breast_Cancer[idx,]

# Loading 30% Breast cancer in test dataset 
test <- Breast_Cancer[-idx, ]

# Preparing & Plotting Model
model <- C5.0(Class~. , training[,-1])
summary(model)
plot(model)

# Predicting Class for test set
prediction<-predict(model,test[,-1],type="class") 
table(test[,11],prediction)
str(prediction)

# Error Rate
wrong<-sum(test[,11]!=prediction)
error_rate<-wrong/length(test[,11])
error_rate


