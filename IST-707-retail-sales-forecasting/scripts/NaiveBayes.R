library(ggplot2)
library(e1071)
library(readr)
library(caret)
library(rattle)
library(MLmetrics)
library(dplyr)
library(FNN)
library(rpart) 
library(rpart.plot)
library(infotheo)
library(randomForest)

# Folder Path
# paths are relative to the project folder
FOLDER_PATH <- ''

# Path
PATH <- paste0(FOLDER_PATH,'data/refined/classification_data.csv')

# read in dataframe
df <- read.csv(PATH)

# id columns to filter on
cols = c("store_id","wm_yr_wk","event_name_1",'sales',
         "event_type_1","event_name_2",
         "week","mday", "best_day",
         "item_nunique_price")

# factorize best day column
df$best_day <- as.factor(df$best_day)
df$event_name_1 <- as.factor(df$event_name_1)
df$event_name_2 <- as.factor(df$event_name_2)
df$event_type_1 <- as.factor(df$event_type_1)
df$mday         <- as.factor(df$mday)
df$week         <- as.factor(df$week)

df$item_nunique_price<- as.factor(df$item_nunique_price)

# create index for training and testing
idx <-createDataPartition(y=df$best_day,p=0.7,list=FALSE)

# setup train and test dataframes
train <- df[idx, cols]
test  <- df[-idx,cols]


# for each laplace variable
for (laplace in c(.0001,.01,.1,1)){
    
    # train model
    model <- naiveBayes(best_day~., data = train, laplace = laplace, na.action = na.pass)
    
    # calculate tran test score
    train_score <- mean(train$best_day == predict(model,train,type="class"))
    test_score  <- mean(test$best_day == predict(model,test,type="class"))
    
    
    # print score
    print(paste0('laplace: ',laplace))
    print(paste0('Train Score: ',train_score))
    print(paste0('Test Score: ',test_score))
    print(' ')
}
