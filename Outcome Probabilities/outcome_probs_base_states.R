# Load in necessary libraries
library(tidyverse)
library(arrow)

# Necessary data (using 2024-25)
data_24 <- read_parquet("pbp_2024.parquet")
data_25 <- read_parquet("pbp_2025.parquet")

data_all <- bind_rows(data_24, data_25)



# Filter to last pitch of each PA & build context variables
# Note: events column is only populated on the final pitch of each PA
pa_data <- data_all %>%
  filter(!is.na(events), events != "") %>%
  mutate(
    outcome = case_when(
      events %in% c("strikeout", "strikeout_double_play")          ~ "Strikeout",
      events %in% c("walk", "intent_walk")                         ~ "Walk",
      events == "hit_by_pitch"                                     ~ "HBP",
      events == "single"                                            ~ "Single",
      events == "double"                                            ~ "Double",
      events == "triple"                                            ~ "Triple",
      events == "home_run"                                          ~ "Home Run",
      events %in% c("field_out", "force_out", "grounded_into_double_play",
                    "double_play", "triple_play", "fielders_choice",
                    "fielders_choice_out", "sac_fly", "sac_bunt",
                    "sac_fly_double_play", "sac_bunt_double_play")  ~ "Other Out",
      TRUE ~ NA_character_
    ),
    
    # Combined count label
    count = paste0(balls, "-", strikes),
    
    # Outs
    outs = factor(outs_when_up),
    
    # Base states (8 of them)
    base_state = case_when(
      is.na(on_1b) & is.na(on_2b) & is.na(on_3b) ~ "Empty",
      !is.na(on_1b) & is.na(on_2b) & is.na(on_3b) ~ "1B",
      is.na(on_1b) & !is.na(on_2b) & is.na(on_3b) ~ "2B",
      is.na(on_1b) & is.na(on_2b) & !is.na(on_3b) ~ "3B",
      !is.na(on_1b) & !is.na(on_2b) & is.na(on_3b) ~ "1B-2B",
      !is.na(on_1b) & is.na(on_2b) & !is.na(on_3b) ~ "1B-3B",
      is.na(on_1b) & !is.na(on_2b) & !is.na(on_3b) ~ "2B-3B",
      !is.na(on_1b) & !is.na(on_2b) & !is.na(on_3b) ~ "Loaded",
      TRUE ~ "Unknown"
    )
  ) %>%
  filter(!is.na(outcome), base_state != "Unknown")

cat("Total PAs for analysis:", nrow(pa_data), "\n")
cat("Outcome distribution:\n")
print(table(pa_data$outcome))


# 1) OUTCOME VS COUNT
count_table <- table(pa_data$outcome, pa_data$count)
chi_count   <- chisq.test(count_table)

cat("\n── Chi-Square: Outcome vs Count ──\n")
cat(sprintf("Chi-Square: %.2f | df: %d | p-value: %e\n",
            chi_count$statistic, chi_count$parameter, chi_count$p.value))

# Outcome proportions by count (to visualize later in code)
count_props <- pa_data %>%
  group_by(count, outcome) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(count) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()


# 2) OUTCOME VS OUTS
outs_table <- table(pa_data$outcome, pa_data$outs)
chi_outs   <- chisq.test(outs_table)

cat("\n── Chi-Square: Outcome vs Outs ──\n")
cat(sprintf("Chi-Square: %.2f | df: %d | p-value: %e\n",
            chi_outs$statistic, chi_outs$parameter, chi_outs$p.value))

# Outcome proportions by outs
outs_props <- pa_data %>%
  group_by(outs, outcome) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(outs) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()


# 3) OUTCOME VS BASE STATE
base_table <- table(pa_data$outcome, pa_data$base_state)
chi_base   <- chisq.test(base_table)

cat("\n── Chi-Square: Outcome vs Base State ──\n")
cat(sprintf("Chi-Square: %.2f | df: %d | p-value: %e\n",
            chi_base$statistic, chi_base$parameter, chi_base$p.value))

# Outcome proportions by base state
base_props <- pa_data %>%
  group_by(base_state, outcome) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(base_state) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()


# 4) CRAMÉR'S V — EFFECT SIZE
# Chi-square is almost guaranteed to be significant with large amount of obs. like we have here
# Cramér's V will tell us whether the association is practically meaningful
# Reference:
# ~0.10 = small, ~0.20 = medium, ~0.30 = large

cramers_v <- function(chi_result, n) {
  k <- min(dim(chi_result$observed))
  sqrt(chi_result$statistic / (n * (k - 1)))
}

n <- nrow(pa_data)

v_count <- cramers_v(chi_count, n)
v_outs  <- cramers_v(chi_outs,  n)
v_base  <- cramers_v(chi_base,  n)

cat("\n── Cramér's V (Effect Size) ──\n")
cat(sprintf("Count:      V = %.4f\n", v_count))
cat(sprintf("Outs:       V = %.4f\n", v_outs))
cat(sprintf("Base State: V = %.4f\n", v_base))

# Summary table
effect_summary <- tibble(
  Variable   = c("Count", "Outs", "Base State"),
  Chi_Square = round(c(chi_count$statistic, chi_outs$statistic, chi_base$statistic), 2),
  df         = c(chi_count$parameter, chi_outs$parameter, chi_base$parameter),
  p_value    = c(chi_count$p.value, chi_outs$p.value, chi_base$p.value),
  Cramers_V  = round(c(v_count, v_outs, v_base), 4)
)
cat("\n── Full Summary ──\n")
print(effect_summary)


# 5) VISUALIZATIONS
cluster_colors <- c("#1b7837", "#4393c3", "#2d9e9e",
                    "#6baed6", "#807dba", "#54278f", "#00798c", "#e08214")

# Order counts logically
count_order <- c("0-0","0-1","0-2","1-0","1-1","1-2","2-0","2-1","2-2","3-0","3-1","3-2")

# Stacked bar: outcome proportions by count
outcome_order <- c("Home Run", "Triple", "Double", "Single", "Walk", "HBP", "Strikeout", "Other Out")


ggplot(count_props %>% mutate(
  count   = factor(count, levels = count_order),
  outcome = factor(outcome, levels = outcome_order)),
  aes(x = count, y = prop, fill = outcome)) +
  geom_col(position = "fill") +
  scale_fill_manual(values = cluster_colors) +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title    = "PA Outcome Distribution by Count",
    subtitle = paste0("Cramér's V = ", round(v_count, 4), " — Strong association"),
    x        = "Count",
    y        = "Proportion of PAs",
    fill     = "Outcome"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.title.y      = element_text(face = "bold", size = 14),
    axis.title.x      = element_text(face = "bold", size = 14),
    plot.title        = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle     = element_text(hjust = 0.5),
    legend.title      = element_text(face = "bold"),
    legend.background = element_rect(fill = "lightyellow", color = "black", linewidth = 0.5)
  )

ggsave("outcome_proportions_count.png", width = 10, height = 6, dpi = 300)


# Stacked bar: outcome proportions by outs
ggplot(outs_props %>% mutate(outcome = factor(outcome, levels = outcome_order)),
       aes(x = outs, y = prop, fill = outcome)) +
  geom_col(position = "fill") +
  scale_fill_manual(values = cluster_colors) +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title    = "PA Outcome Distribution by Outs",
    subtitle = paste0("Cramér's V = ", round(v_outs, 4), " — Weak association"),
    x        = "Outs",
    y        = "Proportion of PAs",
    fill     = "Outcome"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.title.y      = element_text(face = "bold", size = 14),
    axis.title.x      = element_text(face = "bold", size = 14),
    plot.title        = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle     = element_text(hjust = 0.5),
    legend.title      = element_text(face = "bold"),
    legend.background = element_rect(fill = "lightyellow", color = "black", linewidth = 0.5)
  )

ggsave("outcome_proportions_outs.png", width = 10, height = 6, dpi = 300)


# Stacked bar: outcome proportions by base state
base_order <- c("Empty", "1B", "2B", "3B", "1B-2B", "1B-3B", "2B-3B", "Loaded")

ggplot(base_props %>% mutate(
  base_state = factor(base_state, levels = base_order),
  outcome    = factor(outcome, levels = outcome_order)),
  aes(x = base_state, y = prop, fill = outcome)) +
  geom_col(position = "fill") +
  scale_fill_manual(values = cluster_colors) +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title    = "PA Outcome Distribution by Base State",
    subtitle = paste0("Cramér's V = ", round(v_base, 4), " — Weak association"),
    x        = "Base State",
    y        = "Proportion of PAs",
    fill     = "Outcome"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.title.y      = element_text(face = "bold", size = 14),
    axis.title.x      = element_text(face = "bold", size = 14),
    plot.title        = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle     = element_text(hjust = 0.5),
    legend.title      = element_text(face = "bold"),
    legend.background = element_rect(fill = "lightyellow", color = "black", linewidth = 0.5)
  )

ggsave("outcome_proportions_base_states.png", width = 10, height = 6, dpi = 300)


# Save the effect summary as a csv
write_csv(effect_summary, "chi_square_summary.csv")


