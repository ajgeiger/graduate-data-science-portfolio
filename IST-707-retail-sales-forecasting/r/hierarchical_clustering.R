# Hierarchical clustering (Manhattan distance, Ward linkage) on the small
# clustering sample. Run from the project folder after notebooks/data_curation.ipynb.

library(dplyr)
library(ggplot2)

clustering_data <- read.csv("data/refined/clustering_data.csv")

distances <- dist(clustering_data, method = "manhattan")
tree <- hclust(distances, method = "ward.D")
clustering_data$cluster <- cutree(tree, k = 4)

clustering_data %>%
  group_by(cluster) %>%
  summarise(sales = mean(sales), unique_items = mean(item_nunique_price)) %>%
  ggplot(aes(x = cluster, y = sales, fill = unique_items)) +
  geom_bar(position = "dodge", stat = "identity")
