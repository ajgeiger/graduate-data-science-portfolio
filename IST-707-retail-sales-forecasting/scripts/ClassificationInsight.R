library(vroom)
library(readr)
library(stringr)
library(DescTools)
library(ggplot2)
library(plotly)
library(dplyr)
library(timetk)
library(tidyr)
library(ggthemes)

# Special thanks to heads or tails in his r markdown file.
# I modified the following code from this kernel.
# https://www.kaggle.com/headsortails/back-to-predict-the-future-interactive-m5-eda/code

# setup folders and path
# paths are relative to the project folder
folder_root <- ''
path <- str_c(folder_root,'data/raw/')

# import tables
train <- vroom(str_c(path,'sales_train_validation.csv'), delim = ",", col_types = cols())
prices <- vroom(str_c(path,'sell_prices.csv'), delim = ",", col_types = cols())
calendar <- read_csv(str_c(path,'calendar.csv'), col_types = cols())
sample_submit <- vroom(str_c(path,'sample_submission.csv'), delim = ",", col_types = cols())

# define minimum date
min_date <- date("2011-01-29")


# define date function
extract_ts <- function(df){
  
  min_date <- date("2011-01-29")
  
  df %>%
    select(id, starts_with("d_")) %>%  
    pivot_longer(starts_with("d_"), names_to = "dates", values_to = "sales") %>%
    mutate(dates = as.integer(str_remove(dates, "d_"))) %>% 
    mutate(dates = min_date + dates - 1) %>% 
    mutate(id = str_remove(id, "_validation"))
}



foo <- train %>% 
  summarise_at(vars(starts_with("d_")), sum) %>% 
  mutate(id = 1)

bar <- extract_ts(foo) %>% 
  filter(!str_detect(as.character(dates), "-12-25"))

loess_all <- predict(loess(bar$sales ~ as.integer(bar$dates - min(bar$dates)) + 1, span = 1/2, degree = 1))

bar <- bar %>% 
  mutate(loess = loess_all) %>% 
  mutate(sales_rel = sales - loess)

bar %>% 
  ggplot(aes(dates, sales)) +
  geom_line(col = "dodgerblue3", alpha = 0.5) +
  geom_line(aes(dates, loess), col = "black") +
  theme_hc() +
  labs(x = "", y = "Sales", title = "Total Sales with Smoothing Fit + Seasonality in Residuals")

bar %>% 
  mutate(wday = wday(dates, label = TRUE, week_start = 1),
         month = month(dates, label = TRUE),
         year = year(dates)) %>% 
  group_by(wday, month, year) %>% 
  summarise(sales = sum(sales_rel)/1e3) %>%
  ggplot(aes(month, wday, fill = sales)) +
  geom_tile() +
  labs(x = "Month of the year", y = "Day of the week", fill = "Relative Sales [1k]") +
  scale_fill_distiller(palette = "Spectral") +
  theme_hc()

bar %>% 
  mutate(wday = wday(dates, label = TRUE, week_start = 1),
         month = month(dates, label = TRUE),
         year = year(dates)) %>% 
  group_by(wday, month, year) %>% 
  summarise(sales = sum(sales_rel)/1e3) %>%
  ggplot(aes(month, wday, fill = sales)) +
  geom_tile() +
  labs(x = "Month of the year", y = "Day of the week", fill = "Relative Sales [1k]") +
  scale_fill_distiller(palette = "Spectral") +
  theme_hc()

bar %>% 
  mutate(wday = wday(dates, label = TRUE, week_start = 1),
         month = month(dates, label = TRUE),
         year = year(dates)) %>% 
  group_by(wday, month, year) %>% 
  summarise(sales = sum(sales)/1e3) %>%
  ggplot(aes(month, wday, fill = sales)) +
  geom_tile() +
  labs(x = "Month of the year", y = "Day of the week", fill = "Relative Sales [1k]") +
  scale_fill_distiller(palette = "Spectral") +
  theme_hc()