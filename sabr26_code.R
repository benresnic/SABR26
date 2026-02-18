library(tidyverse)
library(arrow)

data_2025 <- read_parquet("pbp_2025.parquet")

colnames(data_2025)

# Assuming your data is in a dataframe called 'data'
# Adjust the dataframe name if needed

library(ggplot2)

# Filter for Freddie Freeman
freeman_data <- data_2025 %>%
  filter(batter_name == "Freddie Freeman" | player_name == "Freddie Freeman")

# Create scatterplot
ggplot(freeman_data, aes(x = attack_angle, y = bat_speed)) +
  geom_point(alpha = 0.6, size = 2) +
  labs(
    title = "Freddie Freeman: Bat Speed vs Attack Angle",
    x = "Attack Angle (degrees)",
    y = "Bat Speed (mph)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

# Pearson correlation
correlation <- cor(freeman_data$bat_speed, freeman_data$attack_angle, use = "complete.obs")
print(correlation)

# More detailed correlation analysis
cor_test <- cor.test(freeman_data$bat_speed, freeman_data$attack_angle)
print(cor_test)

