# HW_09_Cluster
# Nicholas DeRobertis
# I Pledge my Honor That I have Abided by the Stevens Honor System
# CWID: 20006069

# Load the dataset
data <- read.csv("C:/users/nickd/cs513/HW/HW_07_SVM/wisc_bc_ContinuousVar.csv")

# Clean the dataset: Convert diagnosis column to a factor with correct labels and remove the ID column
data$diagnosis <- factor(data$diagnosis, levels = c("B", "M"), labels = c("Benign", "Malignant"))
data <- data[-1]

# Calculate pairwise distances, create a clustering tree, and cut it into two clusters
distance <- dist(data[-1])
cluster <- hclust(distance)
twoclusters <- cutree(cluster, 2)

# Plot the original tree and output the table
plot(cluster)
table(twoclusters, data[, 1])
