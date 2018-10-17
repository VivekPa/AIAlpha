# Set working dir to source dir
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

# Set libPaths.
.libPaths("C:\\Users\\Xue Yao\\.exploratory\\R\\3.5")

# Load required packages.
library(janitor)
library(lubridate)
library(hms)
library(tidyr)
library(stringr)
library(readr)
library(openxlsx)
library(forcats)
library(RcppRoll)
library(dplyr)
library(tibble)
library(exploratory)

# Set OAuth token.
exploratory::setTokenInfo("twitter", as.environment(list(user_id = "3039914070", screen_name = "Axe_Why", oauth_token = "3039914070-PeE1I4paM8Y7OWLAzfLq6lJnIy8if2Boe6ir4gQ", oauth_token_secret = "jfpEjEA17jFqXY16cyDQsNjWI07pJDnXbsCrzD8wf1f3h", consumer_sc = "wqP7VhX5yDEGzLL3eHSbT2wDlJvs4OitruIkd18CQZGzsySFuX")))

# Steps to produce the output
data <- exploratory::select_columns(exploratory::clean_data_frame(exploratory::getTwitter(searchString = '#aapl', n = 20000, lang = '', lastNDays = 20, tokenFileId = '', includeRts = FALSE, withSentiment = TRUE)),"created_at","text","sentiment") %>% readr::type_convert() %>%
  filter(sentiment != 0) %>%
  mutate(date = parse_character(created_at)) %>%
  select(date, everything()) %>%
  separate(date, into = c("date_1", "date"), sep = "\\s+", convert = TRUE) %>%
  select(-date) %>%
  rename(date = date_1) %>%
  select(-created_at)

print (data)
write.csv(data, "twitter.csv", fileEncoding="UTF-8")
