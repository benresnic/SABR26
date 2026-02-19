# =============================================================================
# PVC Simulation Win Probability Heatmap Explorer
# =============================================================================
# Shiny app for visualizing win probability changes based on swing mechanics

library(shiny)
library(bslib)
library(arrow)
library(ggplot2)
library(dplyr)
library(scales)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Function to find the output directory containing simulation parquet files
find_output_dir <- function() {
  candidates <- c(
    file.path(getwd(), "Simulations", "output"),
    file.path(dirname(getwd()), "Simulations", "output"),
    file.path(getwd(), "..", "Simulations", "output")
  )

  for (path in candidates) {
    if (dir.exists(path)) {
      return(normalizePath(path))
    }
  }

  # Return first candidate as default
  return(candidates[1])
}

OUTPUT_DIR <- find_output_dir()

# Get list of available players from parquet files
get_available_players <- function() {
  files <- list.files(OUTPUT_DIR, pattern = "_simulation\\.parquet$")
  players <- gsub("_simulation\\.parquet$", "", files)
  players <- gsub("_", " ", players)
  return(sort(players))
}

# Base state mappings
BASE_STATE_CHOICES <- c(
  "Empty" = 0,
  "Runner on 1B" = 1,
  "Runner on 2B" = 2,
  "Runners on 1B & 2B" = 3,
  "Runner on 3B" = 4,
  "Runners on 1B & 3B" = 5,
  "Runners on 2B & 3B" = 6,
  "Bases Loaded" = 7
)

BASE_STATE_NAMES <- c("empty", "1B", "2B", "1B_2B", "3B", "1B_3B", "2B_3B", "loaded")

# Count choices
COUNT_CHOICES <- c(
  "0-0", "0-1", "0-2",
  "1-0", "1-1", "1-2",
  "2-0", "2-1", "2-2",
  "3-0", "3-1", "3-2"
)

# xFIP percentile choices (lower = better pitcher)
XFIP_CHOICES <- c(
  "10th (Elite)" = 10,
  "30th (Good)" = 30,
  "50th (Average)" = 50,
  "70th (Below Avg)" = 70,
  "90th (Poor)" = 90
)

XFIP_QUALITY <- c(
  "10" = "elite",
  "30" = "good",
  "50" = "avg",
  "70" = "below avg",
  "90" = "poor"
)

# -----------------------------------------------------------------------------
# Plotting Function
# -----------------------------------------------------------------------------
plot_wp_diff_heatmap <- function(player_name, p_count, p_outs, p_base_state,
                                  p_inning, p_inning_topbot, p_score_diff,
                                  p_pitcher_xfip_percentile) {

  # Load player simulation data
  safe_name <- gsub(" ", "_", gsub("\\.", "", player_name))
  file_path <- file.path(OUTPUT_DIR, paste0(safe_name, "_simulation.parquet"))

  if (!file.exists(file_path)) {
    stop(paste("Simulation file not found:", file_path))
  }

  df <- read_parquet(file_path)

  # Filter to specific game state
  state_df <- df %>%
    filter(
      count == p_count,
      outs == p_outs,
      base_state == p_base_state,
      inning == p_inning,
      inning_topbot == p_inning_topbot,
      score_diff == p_score_diff,
      pitcher_xfip_percentile == p_pitcher_xfip_percentile
    )

  if (nrow(state_df) == 0) {
    stop(paste(
      "No data found for state: count=", p_count, ", outs=", p_outs,
      ", base_state=", p_base_state, ", inning=", p_inning,
      ", topbot=", p_inning_topbot, ", score_diff=", p_score_diff,
      ", xfip_pct=", p_pitcher_xfip_percentile
    ))
  }

  # Get baseline values
  baseline_row <- state_df %>%
    filter(bat_speed_change_pct == 0, swing_length_change_pct == 0) %>%
    slice(1)

  baseline_bs <- baseline_row$baseline_bat_speed
  baseline_sl <- baseline_row$baseline_swing_length
  wp_before <- baseline_row$wp_before_pa
  baseline_wp_after <- baseline_row$baseline_wp_after_pa

  # Convert wp_diff to percentage
  state_df <- state_df %>%
    mutate(wp_diff_pct = wp_diff_from_baseline * 100)

  # Base state name
  base_name <- BASE_STATE_NAMES[p_base_state + 1]

  # Score label
  if (p_score_diff < 0) {
    score_label <- paste("down by", abs(p_score_diff))
  } else if (p_score_diff > 0) {
    score_label <- paste("up by", p_score_diff)
  } else {
    score_label <- "tie game"
  }

  # xFIP quality label
  quality_label <- XFIP_QUALITY[as.character(p_pitcher_xfip_percentile)]

  # Create title
  title <- paste0(
    player_name, "\n",
    "State: ", p_count, " count, ", p_outs, " out, ", base_name, ", ",
    ifelse(p_inning_topbot == "Top", "Top", "Bot"), " ", p_inning, ", ",
    score_label, "\n",
    "WP before PA: ", sprintf("%.3f", wp_before),
    " | Baseline WP after: ", sprintf("%.3f", baseline_wp_after),
    " | Pitcher: ", p_pitcher_xfip_percentile, "th pct xFIP (", quality_label, ")"
  )

  # Create the heatmap
  p <- ggplot(state_df, aes(x = new_swing_length, y = new_bat_speed, fill = wp_diff_pct)) +
    geom_tile(color = "white", linewidth = 0.3) +
    scale_fill_gradient2(
      low = "#d73027",
      mid = "#ffffbf",
      high = "#1a9850",
      midpoint = 0,
      limits = c(-1, 1),
      oob = scales::squish,
      name = "WP Change from Baseline (%)",
      guide = guide_colorbar(
        barheight = unit(12, "cm"),
        barwidth = unit(0.5, "cm"),
        title.position = "right",
        title.theme = element_text(angle = 90, hjust = 0.5, vjust = 0.5, size = 11),
        label.position = "left",
        ticks = FALSE
      )
    ) +
    # Add text labels for values outside -1% to +1%
    geom_text(
      data = state_df %>% filter(abs(wp_diff_pct) > 1),
      aes(label = sprintf("%+.1f%%", wp_diff_pct)),
      color = "white",
      size = 3,
      fontface = "bold"
    ) +
    # Mark baseline position
    geom_point(
      data = data.frame(x = baseline_sl, y = baseline_bs),
      aes(x = x, y = y),
      inherit.aes = FALSE,
      shape = 21,
      size = 5,
      fill = "black",
      color = "white",
      stroke = 1.5
    ) +
    # Reference lines
    geom_hline(yintercept = baseline_bs, linetype = "dashed", alpha = 0.4, color = "black") +
    geom_vline(xintercept = baseline_sl, linetype = "dashed", alpha = 0.4, color = "black") +
    # Labels
    labs(
      title = title,
      x = "Adjusted Swing Length (ft)",
      y = "Adjusted Bat Speed (mph)"
    ) +
    theme_bw(base_size = 14) +
    theme(
      aspect.ratio = 1,
      axis.title.y = element_text(face = "bold", size = 14),
      axis.title.x = element_text(face = "bold", size = 14),
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid = element_blank(),
      legend.position = "right",
      axis.text = element_text(size = 10),
      plot.margin = margin(15, 15, 15, 15)
    )

  return(p)
}

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

custom_css <- "
  .shiny-input-container { margin-bottom: 0.2rem !important; }
  .form-label, label.control-label { margin-bottom: 0 !important; font-size: 0.75rem; }
"

ui <- page_sidebar(
  title = NULL,
  theme = bs_theme(
    version = 5,
    bootswatch = "flatly",
    primary = "#0d6efd",
    "sidebar-bg" = "#f8f9fa"
  ),

  tags$head(tags$style(HTML(custom_css))),

  sidebar = sidebar(
    id = "sidebar",
    width = 250,
    open = TRUE,
    padding = "0.5rem",

    selectInput("player", "Player", choices = NULL, width = "100%"),
    fluidRow(
      column(6, selectInput("count", "Count", choices = COUNT_CHOICES, selected = "2-0", width = "100%")),
      column(6, selectInput("outs", "Outs", choices = c("0" = 0, "1" = 1, "2" = 2), selected = 2, width = "100%"))
    ),
    selectInput("base_state", "Bases", choices = BASE_STATE_CHOICES, selected = 6, width = "100%"),
    fluidRow(
      column(6, selectInput("inning", "Inning", choices = c("7" = 7, "8" = 8, "9" = 9), selected = 9, width = "100%")),
      column(6, selectInput("topbot", "Half", choices = c("Top" = "Top", "Bot" = "Bot"), selected = "Bot", width = "100%"))
    ),
    selectInput("score_diff", "Score Diff", choices = setNames(-3:3, c("-3", "-2", "-1", "0", "+1", "+2", "+3")), selected = -1, width = "100%"),
    selectInput("xfip_pct", "Pitcher xFIP", choices = XFIP_CHOICES, selected = 30, width = "100%"),
    actionButton("generate", "Generate", class = "btn-primary w-100 mt-2")
  ),

  plotOutput("heatmap", height = "100%")
)

# -----------------------------------------------------------------------------
# Server
# -----------------------------------------------------------------------------
server <- function(input, output, session) {

  # Populate player choices on startup
  observe({
    players <- get_available_players()
    updateSelectInput(session, "player", choices = players, selected = "Kyle Schwarber")
  })

  # Reactive value to store the plot
  plot_data <- reactiveVal(NULL)

  # Generate plot on button click
  observeEvent(input$generate, {
    req(input$player)

    # Show loading state
    showNotification(
      "Generating heatmap...",
      type = "message",
      duration = 2,
      id = "loading"
    )

    tryCatch({
      p <- plot_wp_diff_heatmap(
        player_name = input$player,
        p_count = input$count,
        p_outs = as.integer(input$outs),
        p_base_state = as.integer(input$base_state),
        p_inning = as.integer(input$inning),
        p_inning_topbot = input$topbot,
        p_score_diff = as.integer(input$score_diff),
        p_pitcher_xfip_percentile = as.integer(input$xfip_pct)
      )
      plot_data(p)
      removeNotification(id = "loading")
    }, error = function(e) {
      removeNotification(id = "loading")
      showNotification(
        paste("Error:", e$message),
        type = "error",
        duration = 5
      )
    })
  })
  # Render the plot
  output$heatmap <- renderPlot({
    p <- plot_data()
    if (is.null(p)) {
      ggplot() + annotate("text", x = 0.5, y = 0.5, label = "Click Generate", size = 5, color = "#999") + theme_void()
    } else {
      p
    }
  }, res = 96)

  # Auto-generate on first load after player selection
  observe({
    req(input$player)
    # Only trigger once on initial load
    isolate({
      if (is.null(plot_data())) {
        # Slight delay to ensure UI is ready
        invalidateLater(500)
      }
    })
  })
}

# -----------------------------------------------------------------------------
# Run App
# -----------------------------------------------------------------------------
shinyApp(ui = ui, server = server)
