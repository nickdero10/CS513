# HW_08_ANN
# Nicholas DeRobertis
# I Pledge my Honor That I have Abided by the Stevens Honor System
# CWID: 20006069

# Clear workspace
rm(list=ls())

# Load required library
library(neuralnet)

# Read data
data <- read.csv("C:/users/nickd/cs513/HW/HW_07_SVM/wisc_bc_ContinuousVar.csv")

# Assigning columns to variables
id <- data[, "id"]
diagnosis <- data[, "diagnosis"]

# Scaling the data
scaled_data <- scale(data[, 3:32])

# Combining id and diagnosis with scaled data
data <- data.frame(id, diagnosis, scaled_data)

# Converting diagnosis to factor
data$diagnosis <- factor(data$diagnosis, levels = c("B", "M"), labels = c("Benign", "Malignant"))

# Splitting data into training and testing sets
split_index <- sample(nrow(data), 0.7 * nrow(data))
training_data <- data[split_index, -1]
testing_data <- data[-split_index, -1]

# Creating the neural network model
ann_model <- neuralnet(data$diagnosis ~ ., data = training_data, hidden = 5)

# Making predictions
predicted_values <- predict(ann_model, testing_data)
final_Predictions = factor(levels = c("Benign","Malignant"), labels = c("Benign", "Malignant"))
for (i in 1:nrow(predicted_values)) {
  if (predicted_values[i,1] >  predicted_values[i,2]) {
    final_Predictions[i] <- "Benign"
  } else {
    final_Predictions[i] <- "Malignant"
  }
}

# Outputting confusion matrix and accuracy
ann_confusion_table <- table(final_Predictions, testing_data$diagnosis)
ann_accuracy <- sum(diag(ann_confusion_table)) / nrow(testing_data)
ann_confusion_table
ann_accuracy

