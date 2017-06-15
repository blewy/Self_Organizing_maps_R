# Load the kohonen package 
require(kohonen)
require(tidyverse)
require(data.table)
# Clustering
library("cluster")
library("factoextra")
library(mlbench)

#------------- Load the Data -------------

data<-read.table("./data/lowbwt.dat", header=TRUE, skip=0)

#--------- Understand the Data -----------

# Low Birthrate data:
#
#Columns    Variable                                              Abbreviation
#-----------------------------------------------------------------------------
# Identification Code                                     ID
# Low Birth Weight (0 = Birth Weight >= 2500g,            LOW
#                          1 = Birth Weight < 2500g)
# Age of the Mother in Years                              AGE
# Weight in Pounds at the Last Menstrual Period           LWT
# Race (1 = White, 2 = Black, 3 = Other)                  RACE
# Smoking Status During Pregnancy (1 = Yes, 0 = No)       SMOKE
# History of Premature Labor (0 = None  1 = One, etc.)    PTL
# History of Hypertension (1 = Yes, 0 = No)               HT
# Presence of Uterine Irritability (1 = Yes, 0 = No)      UI
# Number of Physician Visits During the First Trimester   FTV
#                (0 = None, 1 = One, 2 = Two, etc.)
# Birth Weight in Grams                                   BWT
#------------------------------------------------------------


# Create a training data set (rows are samples, columns are variables
# Here I am selecting a subset of my variables available in "data"
data_train <- data[, c(2:11)]

#-------------- Visualization ------------
par(mfrow=c(2,4))
categorical<- c("LOW","RACE","SMOKE","PTL","HT","UI","FTV")

#Bar-Plot
catseq<-which(colnames(data_train) %in% categorical)
for(i in catseq) {
  counts <- table(data_train[,i])
  name <- names(data_train)[i]
  barplot(counts, main=name)
}

#-------------- Bax-Plot ----------------
continous<- c("AGE","LWT","BWT")
contseq<-which(colnames(data_train) %in% continous)
# Create separate boxplots for each attribute
par(mfrow=c(1,3))
for(i in contseq) {
  boxplot(data_train[,i], main=names(data_train)[i])
}

#-------------- Histogram --------------
par(mfrow=c(1,3))
for(i in contseq) {
  hist(data_train[,i], main=names(data_train)[i])
}

#-------------- Density-plot -----------
par(mfrow=c(1,3))
for(i in contseq) {
  plot(density(data_train[,i]), main=names(data_train)[i])
}

dev.off() # Stop de multiplot

#------------------- Modelling SOM ----------------- 

# Change the data frame with training data to a matrix
# Also center and scale all variables to give them equal importance during
# the SOM training process. 
data_train_matrix <- as.matrix(scale(data_train))

# Create the SOM Grid - you generally have to specify the size of the 
# training grid prior to training the SOM. Hexagonal and Circular 
# topologies are possible
som_grid <- somgrid(xdim = 4, ydim=4, topo="hexagonal")

# Finally, train the SOM, options for the number of iterations,
# the learning rates, and the neighbourhood are available
set.seed(123)
som_model <- som(data_train_matrix, 
                 grid=som_grid, 
                 rlen=1000, 
                 alpha=c(0.05,0.01), 
                 keep.data = TRUE
                 )

## Training Progress:
plot(som_model, type="changes")

## Node Counts
plot(som_model, type="count")

## Neighbour Distance
plot(som_model, type="dist.neighbours")

## Codes / Weight vectors
plot(som_model, type="codes")

coolBlueHotRed <- function(n, alpha = 1) {
  rainbow(n, end=4/6, alpha=alpha)[n:1]
}

## Heatmaps
#View(data)
var <- 2 #define the variable to plot 
plot(som_model, type = "property", property = as.data.frame(som_model$codes)[,var], main=names(as.data.frame(som_model$data))[var] , palette.name=coolBlueHotRed)

#A more intuitive and useful visualisation is of the variable prior to scaling, which involves some R trickery – using the aggregate function to regenerate the variable from the original training set and the SOM node/sample mappings. The result is scaled to the real values of the training variable (in this case, unemployment percent).

var <- 3 #define the variable to plot 
var_unscaled <- aggregate(as.numeric(data_train[,var]), by=list(som_model$unit.classif), FUN=mean, simplify=TRUE)[,2] 
plot(som_model, type = "property", property=var_unscaled, main=names(data_train)[var], palette.name=coolBlueHotRed)

#Clustering
# Loading and preparing data
my_data <- as.data.frame(som_model$codes)
# Compute dissimilarity matrix
d <- dist(my_data, method = "euclidean")
# Hierarchical clustering using Ward's method
res.hc <- hclust(d, method = "ward.D" )
# Verify number of clusters 
library("NbClust")
set.seed(123)
res.nbclust <- NbClust(my_data, distance = "euclidean",
                       min.nc = 2, max.nc = 10, 
                       method = "ward.D", index ="all") 

factoextra::fviz_nbclust(res.nbclust) + theme_minimal()
# Cut tree into optimal groups
grp <- cutree(res.hc, k = 4)
# Visualize
plot(res.hc, cex = 0.6) # plot tree
rect.hclust(res.hc, k = 4, border = 2:5) # add rectangle


# Using facorextra
som_cluster_v2 <- hcut(my_data, k = 4, stand = TRUE)
# Visualize
fviz_dend(som_cluster_v2, rect = TRUE, cex = 0.5)

# Colour palette definition
pretty_palette <- c("#1f77b4", '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2')

# PLot SOM cluster
plot(som_model, type="codes", bgcol = pretty_palette[grp], main = "Clusters") 
add.cluster.boundaries(som_model, grp)

# PLot SOM cluster using Codes
plot(som_model, type="codes", bgcol = pretty_palette[grp], main = "Clusters") 
add.cluster.boundaries(som_model, grp)

#Plot SOM Cluster Based on Property
var <- 3 #define the variable to plot 
var_unscaled <- aggregate(as.numeric(data_train[,var]), by=list(som_model$unit.classif), FUN=mean, simplify=TRUE)[,2] 
plot(som_model, type = "property", property=var_unscaled, main=names(data_train)[var], palette.name=coolBlueHotRed)
add.cluster.boundaries(som_model, grp)

# Clustering using within SS
#
# mydata <- as.data.frame(som_model$codes)
# wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var)) 
# for (i in 2:15) {
#   wss[i] <- sum(kmeans(mydata, centers=i)$withinss)
# }
# plot(wss)
# plot(1:15, wss, type="b", xlab="Number of Clusters",
#      ylab="Within groups sum of squares", main="Within cluster sum of squares (WCSS)"#)
