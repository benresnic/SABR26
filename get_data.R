library(baseballr)
library(tidyverse)
library(arrow)

get_fg_batter_leaders_season <- function(season, qual = 0) {
  baseballr::fg_batter_leaders(
    startseason = season,
    endseason = season,
    qual = qual
  )
}

get_fg_pitcher_leaders_season <- function(season, qual = 0) {
  baseballr::fg_pitcher_leaders(
    startseason = season,
    endseason = season,
    qual = qual
  )
}

build_batter_stats <- function(seasons = c(2024, 2025), qual = 0) {
  season_frames <- lapply(seasons, get_fg_batter_leaders_season, qual = qual)

  bind_rows(season_frames) %>%
    mutate(xISO = xSLG - xAVG) %>%
    select(
      Season,
      xMLBAMID,
      PlayerName,
      xISO,
      AttackAngle,
      `Z-Contact_pct`,
      `O-Swing_pct`,
      AvgBatSpeed
    )
}

build_pitcher_stats <- function(seasons = c(2024, 2025), qual = 0) {
  season_frames <- lapply(seasons, get_fg_pitcher_leaders_season, qual = qual)

  bind_rows(season_frames) %>%
    select(
      Season,
      xMLBAMID,
      xwOBA
    )
}

save_batter_stats <- function(
  seasons = c(2024, 2025),
  output_path = "Data/batter_stats.parquet",
  qual = 0
) {
  stats <- build_batter_stats(seasons = seasons, qual = qual)
  write_parquet(stats, output_path)
  invisible(stats)
}

save_pitcher_stats <- function(
  seasons = c(2024, 2025),
  output_path = "Data/pitcher_stats.parquet",
  qual = 0
) {
  stats <- build_pitcher_stats(seasons = seasons, qual = qual)
  write_parquet(stats, output_path)
  invisible(stats)
}


batter_stats <- save_batter_stats(seasons = c(2024, 2025), output_path = "Data/batter_stats.parquet")
pitcher_stats <- save_pitcher_stats(seasons = c(2024, 2025), output_path = "Data/pitcher_stats.parquet")