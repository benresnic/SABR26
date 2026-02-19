# ***RUN k_means_clustering.R FIRST***

# Loading in necessary libraries
library(tidyverse)
library(cluster)
library(factoextra)
library(ggrepel)
library(clValid)


# 1.) STATISTICAL VALIDATION -----------------------------------------------

# a.) Within-Cluster Sum of Squares (WCSS) & Between/Total SS ratio
# Higher ratio = clusters are compact and well-separated
cat("── WCSS by Cluster ──\n")
print(km$withinss)
cat(sprintf("\nBetween-cluster SS / Total SS: %.3f\n",
            km$betweenss / km$totss))
# Aim for > 0.60; higher the better

# b.) Silhouette Analysis
# Silhouette score per player: how similar they are to their own cluster vs the next nearest cluster. 
# Range: -1 (wrong cluster) to +1 (perfect fit)
sil <- silhouette(km$cluster, dist(features_scaled))

cat("\n── Average Silhouette Width by Cluster ──\n")
sil_summary <- summary(sil)
print(sil_summary$clus.avg.widths)
cat(sprintf("Overall average silhouette width: %.3f\n",
            sil_summary$avg.width))
# > 0.50 = reasonable; > 0.70 = strong

color_palette <- c("#1b7837", "#4393c3", "#2d9e9e",
                  "#6baed6", "#807dba", "#54278f", "#00798c")

fviz_silhouette(sil) +
  scale_fill_manual(values = color_palette, labels = paste("Cluster", 1:7)) +
  scale_color_manual(values = color_palette, labels = paste("Cluster", 1:7)) +
  labs(
    title = "Silhouette Plot: K-Means Clusters (Non-PV Hitters)",
    subtitle = "Negative values indicate potentially misassigned players",
    fill = "Cluster",
    color = "Cluster",
    x = NULL,
    y = "Silhouette Width"
  ) +
  theme_minimal() +
  theme(
    axis.title.y = element_text(face = "bold", size = 14),
    axis.title.x = element_text(face = "bold", size = 14),
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    legend.title = element_text(face = "bold"),
    legend.background = element_rect(fill = "lightyellow", color = "black", linewidth = 0.5)
  )

ggsave("silhouette_validation_plot.png", width = 10, height = 6, dpi = 300)

# c.) Flag low-silhouette players (potential misassignments)
sil_df <- as.data.frame(sil[, 1:3]) %>%
  rename(Cluster = cluster, Neighbor = neighbor, Sil_Width = sil_width) %>%
  mutate(
    PlayerName = hitters_clustered$Name,
    Cluster    = factor(Cluster)
  ) %>%
  left_join(seed_labels, by = "Cluster") %>%
  arrange(Sil_Width)

cat("\n── 20 Lowest Silhouette Scores (most borderline players) ──\n")
print(head(sil_df %>% select(PlayerName, Cluster, Label, Neighbor, Sil_Width), 20))

# d.) Dunn Index
# Ratio of minimum inter-cluster distance to maximum intra-cluster diameter
# Higher = better separated and more compact clusters
dunn_val <- dunn(dist(features_scaled), km$cluster)
cat(sprintf("\nDunn Index: %.4f\n", dunn_val))

# e.) Calinski-Harabasz Index
# Ratio of between-cluster variance to within-cluster variance
# Higher = better defined clusters
ch_index <- function(data, clusters) {
  n <- nrow(data)
  k <- length(unique(clusters))
  overall_mean <- colMeans(data)
  
  between_ss <- sum(sapply(unique(clusters), function(c) {
    pts <- data[clusters == c, ]
    nrow(pts) * sum((colMeans(pts) - overall_mean)^2)
  }))
  
  within_ss <- sum(sapply(unique(clusters), function(c) {
    pts <- data[clusters == c, ]
    cm  <- colMeans(pts)
    sum(apply(pts, 1, function(r) sum((r - cm)^2)))
  }))
  
  (between_ss / (k - 1)) / (within_ss / (n - k))
}

ch <- ch_index(as.data.frame(features_scaled), km$cluster)
cat(sprintf("Calinski-Harabasz Index: %.2f\n", ch))
# No fixed threshold — use comparatively if we test k=6, k=7, k=8



# 2) MORE BASEBALL LOGIC RELATED VALIDATION ----------------------------------

# a.) Cluster profiles on other metrics that were not used in clustering
# wRC+, BABIP, ISO, BB%, HR/FB%, Hard Hit% — do clusters separate on these?
external_metrics <- data_2025 %>%
  filter(PA >= 5) %>%
  transmute(
    Name     = PlayerName,
    wRC_plus = as.numeric(wRC_plus),
    BABIP    = as.numeric(BABIP),
    ISO      = as.numeric(ISO),
    BB_pct   = as.numeric(BB_pct),
    HR_FB    = as.numeric(HR_FB),
    HardHit_pct = as.numeric(HardHit_pct),
    xwOBA    = as.numeric(xwOBA)
  )

cluster_external <- all_clustered %>%
  select(Name, Cluster, Archetype) %>%
  left_join(external_metrics, by = "Name") %>%
  group_by(Cluster, Archetype) %>%
  summarise(
    n            = n(),
    wRC_plus_avg = round(mean(wRC_plus, na.rm = TRUE), 1),
    BABIP_avg    = round(mean(BABIP,    na.rm = TRUE), 3),
    ISO_avg      = round(mean(ISO,      na.rm = TRUE), 3),
    BB_pct_avg   = round(mean(BB_pct,   na.rm = TRUE), 1),
    HR_FB_avg    = round(mean(HR_FB,    na.rm = TRUE), 1),
    HardHit_avg  = round(mean(HardHit_pct, na.rm = TRUE), 1),
    xwOBA_avg    = round(mean(xwOBA,    na.rm = TRUE), 3),
    .groups = "drop"
  ) %>%
  arrange(Cluster)

cat("\n── Cluster Profiles on External Metrics ──\n")
print(cluster_external)

# b.) ANOVA: do clusters differ significantly on external metrics?
# If clusters are meaningful, wRC+/ISO/etc. should differ across groups
cat("\n── ANOVA: Do clusters differ on external metrics? ──\n")
ext_joined <- all_clustered %>%
  select(Name, Cluster) %>%
  left_join(external_metrics, by = "Name")

for (metric in c("wRC_plus", "ISO", "BABIP", "HardHit_pct", "xwOBA")) {
  formula  <- as.formula(paste(metric, "~ Cluster"))
  aov_res  <- aov(formula, data = ext_joined)
  p_val    <- summary(aov_res)[[1]][["Pr(>F)"]][1]
  cat(sprintf("  %-15s p = %.4f %s\n", metric, p_val,
              ifelse(p_val < 0.05, "✓ significant", "✗ not significant")))
}

# c.) Known player spot-check
# Manually verify a handful of well-known players landed in sensible clusters
known_players <- c("Freddie Freeman", "Luis Arraez", "Yordan Alvarez",
                   "Gunnar Henderson", "Trea Turner", "Mookie Betts", "Bryce Harper")

cat("\n── Known Player Cluster Assignments ──\n")
all_clustered %>%
  filter(Name %in% known_players) %>%
  select(Name, K_pct, HR_pct, Contact_pct, Archetype) %>%
  arrange(Archetype) %>%
  print()

# d.) Visualize cluster separation on external metrics
ext_joined_labeled <- ext_joined %>%
  left_join(seed_labels, by = "Cluster") %>%
  mutate(Archetype = all_clustered$Archetype[match(Name, all_clustered$Name)])

cluster_colors <- c("#1b7837", "#4393c3", "#2d9e9e",
                    "#6baed6", "#807dba", "#54278f", "#00798c", "#e08214")

# wRC+ distribution by cluster
ggplot(ext_joined_labeled %>% filter(!is.na(wRC_plus)),
       aes(x = reorder(Archetype, wRC_plus, median), y = wRC_plus, fill = Archetype)) +
  geom_boxplot(alpha = 0.75, outlier.size = 1) +
  geom_hline(yintercept = 100, linetype = "dashed", color = "gray40") +
  scale_fill_manual(values = cluster_colors) +
  coord_flip() +
  labs(title = "wRC+ Distribution by Cluster",
       subtitle = "External validation — not used in clustering",
       x = NULL, y = "wRC+") +
  theme_minimal(base_size = 12) +
  theme(
    legend.position  = "none",
    axis.title.y     = element_text(face = "bold", size = 14),
    axis.title.x     = element_text(face = "bold", size = 14),
    plot.title       = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle    = element_text(hjust = 0.5)
  )

ggsave("wrc_plus_by_cluster.png", width = 10, height = 6, dpi = 300)


# ISO distribution by cluster
ggplot(ext_joined_labeled %>% filter(!is.na(ISO)),
       aes(x = reorder(Archetype, ISO, median), y = ISO, fill = Archetype)) +
  geom_boxplot(alpha = 0.75, outlier.size = 1) +
  scale_fill_manual(values = cluster_colors) +
  coord_flip() +
  labs(title = "ISO Distribution by Cluster",
       subtitle = "External validation — not used in clustering",
       x = NULL, y = "ISO") +
  theme_minimal(base_size = 12) +
  theme(
    legend.position  = "none",
    axis.title.y     = element_text(face = "bold", size = 14),
    axis.title.x     = element_text(face = "bold", size = 14),
    plot.title       = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle    = element_text(hjust = 0.5)
  )

ggsave("iso_by_cluster.png", width = 10, height = 6, dpi = 300)



# 3.) STABILITY VALIDATION --------------------------------------------------

# a.) Bootstrap stability — do clusters hold up across resamples?
set.seed(42)
n_boot    <- 100
stability <- numeric(n_boot)

for (i in 1:n_boot) {
  idx      <- sample(nrow(features_scaled), replace = TRUE)
  boot_dat <- features_scaled[idx, ]
  boot_km  <- kmeans(boot_dat, centers = seed_centers, iter.max = 300)
  # Compare cluster sizes as a rough stability proxy
  stability[i] <- boot_km$betweenss / boot_km$totss
}

cat(sprintf("\n── Bootstrap Stability (n=%d resamples) ──\n", n_boot))
cat(sprintf("Mean Between/Total SS: %.3f\n", mean(stability)))
cat(sprintf("SD:                    %.3f\n", sd(stability)))
# Low SD = clusters are stable across different random samples

# b.) Export validation summary
# write_csv(cluster_external, "cluster_validation_external_2025.csv")
# write_csv(sil_df,           "cluster_silhouette_scores_2025.csv")

