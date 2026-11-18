# Decision tree for the best sales weekday, with a small grid over cp and depth.
# Run from the project folder after notebooks/data_curation.ipynb.

library(caret)
library(rattle)
library(rpart)

weekly <- read.csv("data/refined/classification_data.csv")
weekly$best_day <- as.factor(weekly$best_day)

model_columns <- c(
  "wm_yr_wk", "event_name_1", "sales", "event_type_1", "event_name_2",
  "event_type_2", "snap_CA", "snap_TX", "snap_WI", "sell_price", "lag_7",
  "lag_28", "rmean_7_7", "rmean_28_7", "rmean_7_28", "rmean_28_28", "week",
  "quarter", "mday", "price_nunique_items", "item_nunique_price",
  "item_nunique_weekly", "best_day"
)

train_rows <- createDataPartition(y = weekly$best_day, p = 0.7, list = FALSE)
train <- weekly[train_rows, model_columns]
test <- weekly[-train_rows, model_columns]

accuracy <- function(model, data) {
  mean(data$best_day == predict(model, data, type = "class"))
}

for (cp in c(0.0001, 0.001)) {
  for (max_depth in c(10, 15, 20)) {
    control <- rpart.control(minsplit = 20, cp = cp, maxcompete = max_depth, maxdepth = max_depth)
    model <- rpart(best_day ~ ., data = train, method = "class", control = control)

    print(paste0("cp: ", cp))
    print(paste0("max_depth: ", max_depth))
    print(paste0("Train Score: ", accuracy(model, train)))
    print(paste0("Test Score: ", accuracy(model, test)))
    print(" ")
  }
}

# a shallow tree that is easy to read
control <- rpart.control(minsplit = 20, cp = 0.0001, maxcompete = 2, maxdepth = 2)
shallow_tree <- rpart(best_day ~ ., data = train, method = "class", control = control)
fancyRpartPlot(shallow_tree)
