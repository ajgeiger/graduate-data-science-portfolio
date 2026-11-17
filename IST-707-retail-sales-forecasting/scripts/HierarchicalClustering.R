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


df <- read.csv('data/refined/clustering_data.csv')

# create distance vector using the manhatten distance
distance_e = dist(df,method="manhattan")
hc <- hclust(distance_e,method = 'ward.D')
df$cluster<-cutree(hc, k=4)


df %>%
  group_by(cluster) %>%
  summarise(sales = mean(sales), unique_items=mean(item_nunique_price)) %>%
  ggplot(aes(fill=unique_items,y=sales, x=cluster)) + 
  geom_bar(position="dodge", stat="identity")
