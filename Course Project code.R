library(caret)
library(rpart)
library(randomForest)

set.seed(12321)

fileUrl1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
fileUrl2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(fileUrl1, destfile = "./train.csv", method = "curl")
download.file(fileUrl2, destfile = "./test.csv", method = "curl")

train_set <- read.csv("train.csv", header = TRUE, sep = ",", na.strings=c("",NA))
test_set <- read.csv("test.csv", header = TRUE, sep = ",", na.strings=c("",NA))

dim(train_set)
dim(test_set)

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

train_set <- train_set[,-c(1,2,3,4,5,6,7)]
test_set <- test_set[,-c(1,2,3,4,5,6,7)]

####################################

inTrain <- createDataPartition(train_set$classe, p = 0.6, list = FALSE)
train_model <- train_set[inTrain,]
train_validate <- train_set[-inTrain,]

dim(train_model)
dim(train_validate)

#training <- train_set[sample(nrow(train_model), 100), ]

startTime <- Sys.time()

modFit <- train(classe ~ ., data = train_model, method = "rpart")

runTime <- Sys.time() - startTime
runTime

pred_1 <- predict(modFit, train_validate)
confusionMatrix(pred_1, train_validate$classe)

####################################

startTime <- Sys.time()

modFit_rf <- randomForest(classe ~ ., data = train_model)

runTime <- Sys.time() - startTime
runTime

pred_2 <- predict(modFit_rf, train_validate)
confusionMatrix(pred_2, train_validate$classe)

####################################

startTime <- Sys.time()

Final_modFit_rf <- randomForest(classe ~ ., data = train_set)

runTime <- Sys.time() - startTime
runTime

####################################

final_pred <- predict(Final_modFit_rf, test_set)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(final_pred)