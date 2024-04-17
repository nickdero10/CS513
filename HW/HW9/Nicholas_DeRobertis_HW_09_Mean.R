# HW_09_Mean
# Nicholas DeRobertis
# I Pledge my Honor That I have Abided by the Stevens Honor System
# CWID: 20006069

# Load the dataset
data <- read.csv("C:/users/nickd/cs513/HW/HW_07_SVM/wisc_bc_ContinuousVar.csv")

# Clean the dataset: Convert diagnosis column to a factor with correct labels and remove the ID column
data$diagnosis <- factor(data$diagnosis, levels = c("B", "M"), labels = c("Benign", "Malignant"))
data <- data[-1]

# Apply k-means clustering to data (excluding diagnosis column) with a fixed seed
set.seed(100)
twokmeans <- kmeans(data[-1], centers = 2, nstart = 10)

# Convert cluster predictions into a factor class
PredictedValues <- factor(twokmeans$cluster, levels = c(1, 2), labels = c("Benign", "Malignant"))

# Output the confusion table and accuracy
kmeanscomptable <- table(PredictedValues, data$diagnosis)
kmeansacc <- sum(diag(kmeanscomptable) / nrow(data))
kmeanscomptable
kmeansacc
