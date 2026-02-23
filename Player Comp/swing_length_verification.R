# Loading in necessary libraries
library(tidyverse)
library(arrow)

# Needed data
data_25 <- read_parquet("pbp_2025.parquet")


# Verifying bat speed and swing length on selected videos for the in-slides player comp
# Ryan McMahon example
mcmahon <- data_all %>%
  filter(batter_name == "Ryan McMahon",
         grepl("Bender", player_name),
         pitch_type == "ST",
         balls == 3,
         strikes == 2) %>%
  select(game_date, balls, strikes, pitch_type, release_speed, bat_speed, swing_length, player_name)

print(mcmahon)

# Rafael Devers example
devers <- data_all %>%
  filter(batter_name == "Rafael Devers",
         grepl("Kahnle", player_name),
         pitch_type == "CH",
         balls == 3,
         strikes == 2) %>%
  select(game_date, balls, strikes, pitch_type, release_speed, bat_speed, swing_length, player_name)

print(devers)

