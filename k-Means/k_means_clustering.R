# Loading in necessary libraries
library(tidyverse)
library(arrow)
library(baseballr)
library(ggrepel)
library(cluster)
library(factoextra)


# 1.) Data for the clustering
data_2025 <- fg_batter_leaders(startseason = 2025, endseason = 2025, qual = 1)


# 2.) Build clean metric table (min 5 PA)
hitters <- data_2025 %>%
  filter(PA >= 5) %>%
  transmute(
    Name        = PlayerName,
    PA          = PA,
    K_pct       = as.numeric(K_pct),
    HR_pct      = (HR / PA) * 100,
    Contact_pct = as.numeric(Contact_pct)
  ) %>%
  filter(!is.na(K_pct), !is.na(HR_pct), !is.na(Contact_pct))

cat("Total hitters (>=5 PA):", nrow(hitters), "\n")


# 3.) Define the 30 Power-Variance cohort
pv_cohort <- c(
  "Shohei Ohtani", "Aaron Judge", "Cal Raleigh",
  "Oneil Cruz", "James Wood", "Ryan McMahon", "Riley Greene",
  "Eugenio Suárez", "Elly De La Cruz", "Kyle Schwarber", "Jo Adell",
  "Teoscar Hernández", "Spencer Torkelson", "Lawrence Butler",
  "Randy Arozarena", "Adolis García", "Jazz Chisholm Jr.",
  "Christian Walker", "Michael Busch", "Willy Adames", "Taylor Ward",
  "Rafael Devers", "Brent Rooker", "Zach Neto", "Matt Olson",
  "Marcell Ozuna", "Ian Happ", "Pete Crow-Armstrong",
  "Pete Alonso", "Shea Langeliers"
)

# Check for name mismatches
unmatched <- pv_cohort[!pv_cohort %in% hitters$Name]
if (length(unmatched) > 0) {
  cat("⚠ Unmatched cohort names (check spelling):\n")
  print(unmatched)
}

# Split into PV cohort and everyone else
pv_players <- hitters %>% filter(Name %in% pv_cohort)
hitters_filtered <- hitters %>% filter(!Name %in% pv_cohort)

cat("Power-Variance cohort players found:", nrow(pv_players), "\n")
cat("Players for k-means clustering:", nrow(hitters_filtered), "\n")


# 4.) Scale the 3 features (on non-PV players only)
features <- hitters_filtered %>% select(K_pct, HR_pct, Contact_pct)
features_scaled <- scale(features)

scale_center <- attr(features_scaled, "scaled:center")
scale_sd <- attr(features_scaled, "scaled:scale")


# 5.) Pre-seed centroids at the 7 theoretical corners (excluding High-K / High-HR/ Low-Contact)
p25 <- apply(features_scaled, 2, quantile, probs = 0.25)
p75 <- apply(features_scaled, 2, quantile, probs = 0.75)

# 7 combinations — Power-Variance (High K / High HR / Low Contact) will be cluster 8
seed_centers <- rbind(
  c(p75["K_pct"], p75["HR_pct"], p75["Contact_pct"]),  # High K / High HR / High Contact
  c(p75["K_pct"], p25["HR_pct"], p75["Contact_pct"]),  # High K / Low HR  / High Contact
  c(p75["K_pct"], p25["HR_pct"], p25["Contact_pct"]),  # High K / Low HR  / Low Contact
  c(p25["K_pct"], p75["HR_pct"], p75["Contact_pct"]),  # Low K  / High HR / High Contact
  c(p25["K_pct"], p75["HR_pct"], p25["Contact_pct"]),  # Low K  / High HR / Low Contact
  c(p25["K_pct"], p25["HR_pct"], p75["Contact_pct"]),  # Low K  / Low HR  / High Contact
  c(p25["K_pct"], p25["HR_pct"], p25["Contact_pct"])   # Low K  / Low HR  / Low Contact
)
rownames(seed_centers) <- NULL


# 6.) Run k-Means with k=7, seeded at theoretical corners
set.seed(42)
km <- kmeans(features_scaled, centers = seed_centers, iter.max = 300)

hitters_clustered <- hitters_filtered %>%
  mutate(Cluster = factor(km$cluster))


# 7.) Back-transform centroids to raw scale
centers_raw <- as.data.frame(km$centers) %>%
  mutate(
    K_pct       = K_pct       * scale_sd["K_pct"]       + scale_center["K_pct"],
    HR_pct      = HR_pct      * scale_sd["HR_pct"]      + scale_center["HR_pct"],
    Contact_pct = Contact_pct * scale_sd["Contact_pct"] + scale_center["Contact_pct"],
    Cluster     = factor(1:7)
  ) %>%
  relocate(Cluster) %>%
  mutate(across(where(is.numeric), ~ round(., 2)))

cat("\n── Cluster Centroids (raw scale) ──\n")
print(centers_raw)


# 8.) Label clusters directly from seed corner definitions
# This avoids median-drift causing duplicate labels
seed_labels <- tibble(
  Cluster   = factor(1:7),
  Label = c(
    "High K / High HR / High Contact",
    "High K / Low HR / High Contact",
    "High K / Low HR / Low Contact",
    "Low K / High HR / High Contact",
    "Low K / High HR / Low Contact",
    "Low K / Low HR / High Contact",
    "Low K / Low HR / Low Contact"
  )
)

centers_labeled <- centers_raw %>%
  left_join(seed_labels, by = "Cluster")

cat("\n── Cluster Labels ──\n")
print(centers_labeled %>% select(Cluster, K_pct, HR_pct, Contact_pct, Label))


# 9.) Assign descriptive archetype names
archetype_map <- tribble(
  ~Label,                                       ~Archetype,
  "High K / High HR / High Contact",            "Whiff & Barrel",
  "High K / Low HR / High Contact",             "Aggressive Contact",
  "High K / Low HR / Low Contact",              "Free Swinger",
  "Low K / High HR / High Contact",             "Five-Tool Masher",
  "Low K / High HR / Low Contact",              "Pull-Side Power",
  "Low K / Low HR / High Contact",              "Table Setter",
  "Low K / Low HR / Low Contact",               "Slap Hitter"
)

hitters_clustered <- hitters_clustered %>%
  left_join(centers_labeled %>% select(Cluster, Label), by = "Cluster") %>%
  left_join(archetype_map, by = "Label") %>%
  mutate(Archetype = coalesce(Archetype, paste("Cluster", Cluster)))


# 10.) Assign each PV player to their nearest k-means cluster
# Scale PV players using the same center/sd as the non-PV features
pv_scaled <- scale(
  pv_players %>% select(K_pct, HR_pct, Contact_pct),
  center = scale_center,
  scale  = scale_sd
)

# Euclidean distance from each PV player to each of the 7 centroids
dist_to_centers <- as.matrix(dist(rbind(km$centers, pv_scaled)))[
  (nrow(km$centers) + 1):(nrow(km$centers) + nrow(pv_scaled)),
  1:nrow(km$centers)
]

nearest_cluster <- apply(dist_to_centers, 1, which.min)

pv_with_nearest <- pv_players %>%
  mutate(
    Nearest_Cluster = factor(nearest_cluster),
    Nearest_Label   = seed_labels$Label[nearest_cluster],
    Nearest_Archetype = archetype_map$Archetype[
      match(Nearest_Label, archetype_map$Label)
    ]
  )

cat("\n── Power-Variance Players: Nearest Non-PV Cluster ──\n")
print(pv_with_nearest %>%
        select(Name, K_pct, HR_pct, Contact_pct,
               Nearest_Cluster, Nearest_Label, Nearest_Archetype) %>%
        arrange(Nearest_Cluster))

# Add PV cohort as Cluster 8 for the full combined dataset
pv_players_labeled <- pv_players %>%
  mutate(
    Cluster   = factor(8),
    Label = "High K / High HR / Low Contact",
    Archetype = "Power-Variance"
  )

# Combine all players
all_clustered <- bind_rows(hitters_clustered, pv_players_labeled)


# 11.) Final clean dataframe ─────────────────────────────────────────────
final_df <- all_clustered %>%
  transmute(
    xMLBAMID      = data_2025$xMLBAMID[match(Name, data_2025$PlayerName)],
    PlayerName    = Name,
    season        = data_2025$Season[match(Name, data_2025$PlayerName)],
    cluster_idx   = as.numeric(Cluster),
    cluster_label = Label
  )

cat("\n── Final Cluster Dataframe ──\n")
print(final_df)


# 12.) Cluster summary table
cluster_summary <- all_clustered %>%
  group_by(Cluster, Archetype) %>%
  summarise(
    n = n(),
    K_pct_avg = round(mean(K_pct), 1),
    HR_pct_avg = round(mean(HR_pct), 2),
    Contact_pct_avg = round(mean(Contact_pct), 1),
    .groups = "drop"
  ) %>%
  arrange(Cluster)

cat("\n── Full Cluster Summary (7 + Power-Variance) ──\n")
print(cluster_summary)


# 12.) Main scatter: K% vs HR%, Contact% as size, PV highlighted
ggplot(all_clustered, aes(x = K_pct, y = HR_pct,
                          color = Archetype, size = Contact_pct)) +
  geom_point(alpha = 0.75) +
  geom_text_repel(aes(label = Name), size = 2.3, max.overlaps = 15,
                  show.legend = FALSE) +
  scale_size_continuous(range = c(2, 6), name = "Contact %") +
  labs(
    title    = "Hitter Archetypes — 7-Cluster K-Means + Power-Variance Cohort (2025)",
    subtitle = sprintf("n = %d total players, min 5 PA", nrow(all_clustered)),
    x = "K%", y = "HR% (HR/PA)"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "right")


# 13.) Faceted by Contact tier
all_clustered <- all_clustered %>%
  mutate(Contact_bin = cut(Contact_pct,
                           breaks = quantile(Contact_pct, probs = c(0, 1/3, 2/3, 1), na.rm = TRUE),
                           labels = c("Low Contact (Bottom 33%)", "Mid Contact", "High Contact (Top 33%)"),
                           include.lowest = TRUE))

ggplot(all_clustered, aes(x = K_pct, y = HR_pct, color = Archetype)) +
  geom_point(size = 2.8, alpha = 0.85) +
  geom_text_repel(aes(label = Name), size = 2, max.overlaps = 10,
                  show.legend = FALSE) +
  facet_wrap(~ Contact_bin) +
  labs(
    title = "Clusters Faceted by Contact % Tier",
    subtitle = "Tertile splits — shows separation across all 3 dimensions",
    x = "K%", y = "HR% (HR/PA)"
  ) +
  theme_minimal(base_size = 12)


# 14.) Write csv
write_csv(final_df, "hitter_clusters.csv")
# write_csv(cluster_summary, "cluster_summary_2025.csv")
# write_csv(centers_labeled, "cluster_centroids_2025.csv")


