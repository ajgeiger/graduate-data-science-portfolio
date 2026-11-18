# k-nearest neighbors for the best sales weekday.
# Run from the project folder after notebooks/data_curation.ipynb.

library(caret)
library(class)

weekly <- read.csv("data/refined/classification_data.csv")
weekly$best_day <- as.factor(weekly$best_day)

feature_columns <- c(
  "wm_yr_wk", "event_name_1", "sales", "event_type_1", "event_name_2",
  "event_type_2", "snap_CA", "snap_TX", "snap_WI", "sell_price", "lag_7",
  "lag_28", "rmean_7_7", "rmean_28_7", "rmean_7_28", "rmean_28_28", "week",
  "quarter", "mday", "price_nunique_items", "item_nunique_price",
  "item_nunique_weekly"
)

train_rows <- createDataPartition(y = weekly$best_day, p = 0.7, list = FALSE)
train_features <- weekly[train_rows, feature_columns]
test_features <- weekly[-train_rows, feature_columns]
train_labels <- weekly$best_day[train_rows]
test_labels <- weekly$best_day[-train_rows]

for (k in c(4, 5)) {
  predicted <- knn(train = train_features, test = test_features, cl = train_labels, k = k)

  print(paste0("k: ", k))
  print(paste0("Test Score: ", mean(test_labels == predicted)))
  print(" ")
}
