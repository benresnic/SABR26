library(tidyverse)
library(arrow)

data_2025 <- read_parquet("pbp_2025.parquet")

colnames(data_2025)
