# Coefficient plot: bat speed + swing length effects across outcomes

library(arrow)
library(dplyr)
library(ggplot2)
library(broom)

# -----------------------------
# 1) Read + prep
# -----------------------------
batter_stats <- read_parquet("Data/batter_stats.parquet") %>% as_tibble()

pbp <- bind_rows(
  read_parquet("Data/pbp_2024.parquet"),
  read_parquet("Data/pbp_2025.parquet")
)

season_swing_length <- pbp %>%
  group_by(Season = game_year, xMLBAMID = batter) %>%
  summarise(swing_length = mean(swing_length, na.rm = TRUE), .groups = "drop") %>%
  mutate(swing_length = ifelse(is.nan(swing_length), NA_real_, swing_length))

to_prop <- function(x) {
  if (max(x, na.rm = TRUE) > 1) {
    x / 100
  } else {
    x
  }
}

clip01 <- function(x, eps = 1e-4) pmin(pmax(x, eps), 1 - eps)

dat <- batter_stats %>%
  left_join(season_swing_length, by = c("Season", "xMLBAMID")) %>%
  transmute(
    xISO = as.numeric(xISO),
    chase_pct = as.numeric(`O-Swing_pct`),
    z_contact_pct = as.numeric(`Z-Contact_pct`),
    bat_speed = as.numeric(AvgBatSpeed),
    swing_length = as.numeric(swing_length)
  ) %>%
  filter(
    !is.na(xISO), !is.na(chase_pct), !is.na(z_contact_pct),
    !is.na(bat_speed), !is.na(swing_length)
  ) %>%
  mutate(
    chase_logit = qlogis(clip01(to_prop(chase_pct))),
    z_contact_logit = qlogis(clip01(to_prop(z_contact_pct))),
    bat_speed_std = as.numeric(scale(bat_speed)),
    swing_length_std = as.numeric(scale(swing_length))
  )

# -----------------------------
# 2) Fit models
# -----------------------------
m_xiso <- lm(xISO ~ bat_speed_std + swing_length_std, data = dat)
m_zc <- lm(z_contact_logit ~ bat_speed_std + swing_length_std, data = dat)
m_chase <- lm(chase_logit ~ bat_speed_std + swing_length_std, data = dat)

# -----------------------------
# 3) Build coefficient dataframe
# -----------------------------
coef_df <- bind_rows(
  tidy(m_xiso, conf.int = TRUE) %>% mutate(outcome = "xISO"),
  tidy(m_zc, conf.int = TRUE) %>% mutate(outcome = "Z-Contact%"),
  tidy(m_chase, conf.int = TRUE) %>% mutate(outcome = "Chase% / O-Swing%")
) %>%
  filter(term %in% c("bat_speed_std", "swing_length_std")) %>%
  mutate(
    term = recode(term,
      bat_speed_std = "Bat Speed",
      swing_length_std = "Swing Length"
    )
  )

# -----------------------------
# 4) Plot and save (only output)
# -----------------------------
coef_plot <- ggplot(coef_df, aes(x = estimate, y = term, xmin = conf.low, xmax = conf.high, color = term)) +
  geom_vline(xintercept = 0, linetype = 2, alpha = 0.6) +
  geom_pointrange(size = 0.8) +
  facet_wrap(~ outcome, scales = "free_x") +
  scale_color_manual(values = c("Bat Speed" = "#F76900", "Swing Length" = "#000E54")) +
  labs(
    x = "Coefficient Estimate",
    y = NULL,
    title = "Predictor Effects Across Outcomes (Z-Scores)",
    color = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.title.y = element_text(face = "bold", size = 14),
    axis.title.x = element_text(face = "bold", size = 14),
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    panel.background = element_rect(fill = "white"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top"
  )

ggsave(
  filename = "/Users/ameershah/Desktop/SABR26/joint_coef_plot.png",
  plot = coef_plot,
  width = 11,
  height = 5.5,
  dpi = 300
)

cat("Saved visual: /Users/ameershah/Desktop/SABR26/joint_coef_plot.png\n")
cat("Predictors are standardized internally (1 SD units).\n")
cat("Rate outcomes are modeled on logit scale.\n")
