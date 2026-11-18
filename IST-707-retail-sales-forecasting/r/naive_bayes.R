# Naive Bayes for the best sales weekday, trying a few Laplace smoothing values.
# Run from the project folder after notebooks/data_curation.ipynb.

library(caret)
library(e1071)

weekly <- read.csv("data/refined/classification_data.csv")

model_columns <- c(
  "store_id", "wm_yr_wk", "event_name_1", "sales", "event_type_1",
  "event_name_2", "week", "mday", "best_day", "item_nunique_price"
)

# treat these as categories, not numbers
factor_columns <- c(
  "best_day", "event_name_1", "event_name_2", "event_type_1", "mday", "week",
  "item_nunique_price"
)
weekly[factor_columns] <- lapply(weekly[factor_columns], as.factor)

train_rows <- createDataPartition(y = weekly$best_day, p = 0.7, list = FALSE)
train <- weekly[train_rows, model_columns]
test <- weekly[-train_rows, model_columns]

accuracy <- function(model, data) {
  mean(data$best_day == predict(model, data, type = "class"))
}

for (laplace in c(0.0001, 0.01, 0.1, 1)) {
  model <- naiveBayes(best_day ~ ., data = train, laplace = laplace, na.action = na.pass)

  print(paste0("laplace: ", laplace))
  print(paste0("Train Score: ", accuracy(model, train)))
  print(paste0("Test Score: ", accuracy(model, test)))
  print(" ")
}
