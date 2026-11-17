# Total sales trend and weekday by month seasonality.
# Adapted from the "Back to (predict) the future" M5 EDA notebook by Heads or Tails:
# https://www.kaggle.com/headsortails/back-to-predict-the-future-interactive-m5-eda
# Run from the project folder.

library(dplyr)
library(ggplot2)
library(ggthemes)
library(lubridate)
library(stringr)
library(tidyr)
library(vroom)

sales <- vroom("data/raw/sales_train_validation.csv", delim = ",", col_types = cols())
first_date <- as_date("2011-01-29")

# long format: one row per id and date
to_daily_series <- function(wide_sales) {
  wide_sales %>%
    select(id, starts_with("d_")) %>%
    pivot_longer(starts_with("d_"), names_to = "date", values_to = "sales") %>%
    mutate(date = first_date + as.integer(str_remove(date, "d_")) - 1) %>%
    mutate(id = str_remove(id, "_validation"))
}

daily_totals <- sales %>%
  summarise(across(starts_with("d_"), sum)) %>%
  mutate(id = "total") %>%
  to_daily_series() %>%
  filter(!str_detect(as.character(date), "-12-25"))

# loess trend, then sales relative to the trend
days_since_start <- as.integer(daily_totals$date - min(daily_totals$date)) + 1
trend <- predict(loess(daily_totals$sales ~ days_since_start, span = 1 / 2, degree = 1))

daily_totals <- daily_totals %>%
  mutate(trend = trend, relative_sales = sales - trend)

daily_totals %>%
  ggplot(aes(date, sales)) +
  geom_line(col = "dodgerblue3", alpha = 0.5) +
  geom_line(aes(date, trend), col = "black") +
  theme_hc() +
  labs(x = "", y = "Sales", title = "Total sales with loess trend")

by_weekday_month <- daily_totals %>%
  mutate(
    weekday = wday(date, label = TRUE, week_start = 1),
    month = month(date, label = TRUE),
    year = year(date)
  ) %>%
  group_by(weekday, month, year)

by_weekday_month %>%
  summarise(relative_sales = sum(relative_sales) / 1e3, .groups = "drop") %>%
  ggplot(aes(month, weekday, fill = relative_sales)) +
  geom_tile() +
  labs(x = "Month of the year", y = "Day of the week", fill = "Relative Sales [1k]") +
  scale_fill_distiller(palette = "Spectral") +
  theme_hc()

by_weekday_month %>%
  summarise(sales = sum(sales) / 1e3, .groups = "drop") %>%
  ggplot(aes(month, weekday, fill = sales)) +
  geom_tile() +
  labs(x = "Month of the year", y = "Day of the week", fill = "Sales [1k]") +
  scale_fill_distiller(palette = "Spectral") +
  theme_hc()
