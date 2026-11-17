
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
cols = c("wm_yr_wk","event_name_1",'sales', "event_type_1","event_name_2",
         "event_type_2","snap_CA","snap_TX", "snap_WI","sell_price","lag_7",
         "lag_28","rmean_7_7", "rmean_28_7","rmean_7_28","rmean_28_28","week","quarter" ,        
         "mday","price_nunique_items", "item_nunique_price" , "item_nunique_weekly", "best_day")


# factorize best day column
df$best_day <- as.factor(df$best_day)



# create index for training and testing
idx <-createDataPartition(y=df$best_day,p=0.7,list=FALSE)

# setup train and test dataframes
train <- df[idx, cols]
test  <- df[-idx,cols]


# for each cp
for (cp in c(.0001,.001)){
  
  # for each max depth
  for (max_depth in c(10,15,20)){
    
    # setup control dataframe
    control <- rpart.control(minsplit = 20, cp = cp,  maxcompete = max_depth, maxdepth = max_depth)
    
    # train model
    model <- rpart(formula = best_day ~ .,data=train,method="class", control = control)
    
    # calculate tran test score
    train_score <- mean(train$best_day == predict(model,train,type="class"))
    test_score  <- mean(test$best_day == predict(model,test,type="class"))
    
    
    # print score
    print(paste0('cp: ',cp))
    print(paste0('max_depth: ',max_depth))
    print(paste0('Train Score: ',train_score))
    print(paste0('Test Score: ',test_score))
    print(' ')
  }
}


# setup control dataframe
control <- rpart.control(minsplit = 20, cp = .0001,  maxcompete = 2, maxdepth = 2)

# train model
model <- rpart(formula = best_day ~ .,data=train,method="class", control = control)

# train model
fancyRpartPlot(model)
