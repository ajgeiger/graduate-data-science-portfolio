
# THE MODEL WAS TRAINED IN PYTHON (Used Proprietary Features)
# results were imported into R for analysis.
# These results were used in the PPT sides.



# Folder Path
# paths are relative to the project folder
FOLDER_PATH <- ''

# Path
PATH <- paste0(FOLDER_PATH,'data/clusters/clusters_pretrained.csv')

# read in dataframe
data <- read.csv(PATH)




library(ggplot2)
library(dplyr) 

data %>%
  group_by(Cluster) %>%
  summarise(avg_unique_items = mean(item_nunique),std_item_nunique=sd(item_nunique)) %>%
  ggplot(aes(Cluster, avg_unique_items)) + 
  geom_bar(aes(fill = std_item_nunique), position = "dodge", stat="identity")


data %>%
  group_by(Cluster) %>%
  summarise(sales = mean(sales),rolling_sales_mean=mean(rolling_mean_tmp_1_60)) %>%
  ggplot(aes(Cluster, sales)) + 
  geom_bar(aes(fill = rolling_sales_mean), position = "dodge", stat="identity")
