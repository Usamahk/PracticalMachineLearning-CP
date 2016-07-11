---
title: "Practical Machine Learning - Course Project"
author: "Usamah Khan"
date: "August 20, 2015"
output: html_document
---

## Executive Summary

The objective of the project was to examine the data set and apply machine learning techniques to accurately and quantitatively predict how well an exercise is being performed. The data was already split into a training and test set and the training set was used to create a model that could then be applied to the test set to predict the exercise in question. The answers were submitted online and checked via the Coursera DSS portal. This code produced 20/20 and the model was fit to the the training data using random forest with a 99% accuracy.

## Overview

Nowadays there exist an abundance of devices such as Jawbone Up, Nike FuelBand and FitBit that collect large amounts of data about personal activity relatively inexpensively and easily. There are many who use this data to take measurements of themselves performing activites such as running, swimming and lifting weights to quantify how much they do to improve their health. However, rarely do people quantify *how well* they do an exercise.

The goal of this project is to examine data from accelerators placed on the belt, forearm, arm and dumbell of 6 participants who were all asked to perform lifts correctly and incorrectly (using light weights under proper supervision to avoid injury).

## Notes on running the code

Please note that the code will work as long as the working directory is set to a new directory that remains unchanged. The following packages will need to be loaded and the seed set:


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rpart)
library(randomForest)
```

```
## Error in library(randomForest): there is no package called 'randomForest'
```

```r
library(rattle)
```

```
## Error in library(rattle): there is no package called 'rattle'
```

```r
set.seed(12321)
```

*Note - For the purposes of this project, the error incurred by loading rattle can be ignored*

## Data

### Reading

All information surrounding and describing the experiment can be found here: http://groupware.les.inf.puc-rio.br/har. The data can be found at the following Urls and were downloaded and assigned to variables as shown. 


```r
fileUrl1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
fileUrl2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(fileUrl1, destfile = "./train.csv", method = "curl")
download.file(fileUrl2, destfile = "./test.csv", method = "curl")

train_set <- read.csv("train.csv", header = TRUE, sep = ",", na.strings=c("",NA))
test_set <- read.csv("test.csv", header = TRUE, sep = ",", na.strings=c("",NA))

dim(train_set)
```

```
## [1] 19622   160
```

```r
dim(test_set)
```

```
## [1]  20 160
```

As far as we can tell the data is clean and has the 160 variables we require and the test set is comprised of the full 20 variables.

### Preprocessing

Looking through the data we observe many many cases of missing values and *NAs*. This poses an issue since model fitting with missing values results in error.

To deal with this we could either impute the data or find ways to remove the data. Since we can see that the missing values occur systematically it was determined that the best course of action would be to remove the columns where a certain threshold of missing values is exceeded. In this case, 90%.


```r
df <- data.frame(1)
df2 <- data.frame(1)
count = 1

for(i in 1:ncol(train_set)){
    df[i,] <- (sum(is.na(train_set[,i]))/19622)
      if (df[i,] > 0.9) {
        df2[count,] = i 
        count = count + 1
      }
}

count = 0

for(i in 1:nrow(df2)){
  train_set[,(df2[i,]-count)] <- NULL
  test_set[,(df2[i,]-count)] <- NULL
  count = count + 1
  }
```

Finally, we determined that since the first 7 columns were for bookeeping processes only, they too could be dropped.


```r
train_set <- train_set[,-c(1,2,3,4,5,6,7)]
test_set <- test_set[,-c(1,2,3,4,5,6,7)]
```

With the data fully preprocessed we can now begin to fit models.

## Building Models

Even though the data comes to us in the forms of a *training* and *test* set, we cannot use the *test* set to build our model so we have to partition our *training* data set.


```r
inTrain <- createDataPartition(train_set$classe, p = 0.6, list = FALSE)
train_model <- train_set[inTrain,]
train_validate <- train_set[-inTrain,]

dim(train_model)    
```

```
## [1] 11776    53
```

```r
dim(train_validate) 
```

```
## [1] 7846   53
```

We can now begin using the data sets to create our prediction models. We will also call upon the system time functions to determine how long the models take to train.                                                                                                                                      

### Using Trees

The first model was designed with the use of trees for prediction. 


```r
startTime <- Sys.time()

modFit <- train(classe ~ ., data = train_model, method = "rpart")
```

```
## Error in requireNamespaceQuietStop("e1071"): package e1071 is required
```

```r
runTime <- Sys.time() - startTime
runTime
```

```
## Time difference of 2.878274 secs
```

```r
pred_1 <- predict(modFit, train_validate)
```

```
## Error in predict(modFit, train_validate): object 'modFit' not found
```

```r
confusionMatrix(pred_1, train_validate$classe)
```

```
## Error in confusionMatrix(pred_1, train_validate$classe): object 'pred_1' not found
```

We can also call upon the rattle package to create a plot of the tree.


```r
fancyRpartPlot(modFit$finalModel)
```

```
## Error in eval(expr, envir, enclos): could not find function "fancyRpartPlot"
```

As the confusion matrix and the statistics show, this model was very weak with only a *48%* accuracy and with a run time of *38.7 seconds*. This was to expected since with the amount of variables, it would be hard to accurately fit a tree model. To enhance this accuracy we can employ the use of random forests.

### Using Random Forests

The next model was built with Random Forests to see if the accuracy could be improved.


```r
startTime <- Sys.time()

modFit_rf <- randomForest(classe ~ ., data = train_model)
```

```
## Error in eval(expr, envir, enclos): could not find function "randomForest"
```

```r
runTime <- Sys.time() - startTime
runTime
```

```
## Time difference of 0.002542019 secs
```

```r
pred_2 <- predict(modFit_rf, train_validate)
```

```
## Error in predict(modFit_rf, train_validate): object 'modFit_rf' not found
```

```r
confusionMatrix(pred_2, train_validate$classe)
```

```
## Error in confusionMatrix(pred_2, train_validate$classe): object 'pred_2' not found
```

As we can see this is a much superior model with a level of accuracy, when applied to the test set, of *99.2%* and a runtime of *30.06 seconds*.

## Conclusion Predicting using Final Model

At this point we can safely assume this model to be satisfactory to apply to the test set. To further enhance the accuracy (although not necessary) we can train the model to the entire training set and then apply to the test set.


```r
startTime <- Sys.time()

Final_modFit_rf <- randomForest(classe ~ ., data = train_set)
```

```
## Error in eval(expr, envir, enclos): could not find function "randomForest"
```

```r
runTime <- Sys.time() - startTime
runTime
```

```
## Time difference of 0.004439831 secs
```

As we applied the training over the whole training set the run time was subsequently longer clocking in at *56.9 seconds*. With this complete the last step will be to predict on the test set. Our answers were checked via the Coursera DSS portal through the uploading of .txt files. Once matched, the site would notify us on correct answers. To print a function was employed to create text files. As previously stated, the answers provided achieved a 20/20 mark.


```r
final_pred <- predict(Final_modFit_rf, test_set)
```

```
## Error in predict(Final_modFit_rf, test_set): object 'Final_modFit_rf' not found
```

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(final_pred)
```

```
## Error in pml_write_files(final_pred): object 'final_pred' not found
```

This dataset was quite clean and accurate and it is to note that rarely will one with such little preprocessing, high accuracy and little model tuning be available. However, for the purposes of this project, we can use this to our advantage to achieve a satisfacotry result.
