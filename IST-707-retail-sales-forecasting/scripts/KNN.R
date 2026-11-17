
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
library(class)

# Folder Path
# paths are relative to the project folder
FOLDER_PATH <- ''

# Path
PATH <- paste0(FOLDER_PATH,'data/refined/classification_data.csv')

# read in dataframe
df <- read.csv(PATH)


# id columns to filter on
cols = c("wm_yr_wk","event_name_1",'sales', "event_type_1","event_name_2",
         "event_type_2","snap_CA","snap_TX", "snap_WI","sell_price","lag_7",
         "lag_28","rmean_7_7", "rmean_28_7","rmean_7_28","rmean_28_28","week","quarter" ,        
         "mday","price_nunique_items", "item_nunique_price" , "item_nunique_weekly")


# factorize best day column
df$best_day <- as.factor(df$best_day)


# create index for training and testing
idx <-createDataPartition(y=df$best_day,p=0.7,list=FALSE)

# setup train and test dataframes
train_x <- as.data.frame(df[idx, cols])
test_x  <- as.data.frame(df[-idx,cols])

# setup train and test dataframes
train_y <- df[idx,]$best_day
test_y  <- df[-idx,]$best_day


# for each max depth
for (k in c(4,5)){
  
  # results
  test_yhat <- class::knn(train=train_x,test=test_x, cl=train_y, k=k)
  
  
  # calculate tran test score
  test_score  <- mean(test_y$best_day == test_yhat)
  
  
  # print score
  print(paste0('cp: ',cp))
  print(paste0('k: ',k))
  print(paste0('Test Score: ',test_score))
  print(' ')
}


