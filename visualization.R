setwd("/Users/student/Documents/BMI206/bmi203-final/output")
df = read.table("ParameterSearchResults_1.tsv", sep = "\t")
names(df) = c("activationFunction","NumEpochs","BatchSize",
                "OptimizationMetric","learningRate","Score")

library(ggplot2)

# Scatterplot of epochs, colored by activation function
ggplot(df, aes(x = NumEpochs, y = Score, color = activationFunction)) + 
  geom_point(position = position_dodge(width = 0.4))

# Scatterplot of epochs, colored by optimization metric
ggplot(df, aes(x = NumEpochs, y = Score, color = OptimizationMetric)) + 
  geom_point(position = position_dodge(width = 0.4))

# Scatterplot of batch sizes, colored by activation function
ggplot(df, aes(x = BatchSize, y = Score, color = activationFunction)) + 
  geom_point(position = position_dodge(width = 0.4))
