# Plots for the k-means clusters. The clusters were fit in Python and saved
# to data/clusters/clusters_pretrained.csv; these charts went into the slides.
# Run from the project folder.

library(dplyr)
library(ggplot2)

clusters <- read.csv("data/clusters/clusters_pretrained.csv")

clusters %>%
  group_by(Cluster) %>%
  summarise(mean_unique_items = mean(item_nunique), sd_unique_items = sd(item_nunique)) %>%
  ggplot(aes(Cluster, mean_unique_items)) +
  geom_bar(aes(fill = sd_unique_items), position = "dodge", stat = "identity")

clusters %>%
  group_by(Cluster) %>%
  summarise(sales = mean(sales), rolling_mean_sales = mean(rolling_mean_tmp_1_60)) %>%
  ggplot(aes(Cluster, sales)) +
  geom_bar(aes(fill = rolling_mean_sales), position = "dodge", stat = "identity")
