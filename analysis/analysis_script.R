# =============================================================================
# CONFIGURATION — edit everything here, touch nothing below
# =============================================================================
CFG <- list(

  # ── Paths ──────────────────────────────────────────────────────────────────
  # Experiment folder (contains design_matrix/ and experiment_data/ sub-dirs)
  base_dir     = "/Users/rezek_zhu/VLM_Mar18/experiment/Doasisfull_Mb32datacomp_Smin_pairwise",

  # Where to write all outputs; NULL → auto-derive as <results>/<experiment_id>
  out_dir      = NULL,

  # OASIS ratings file (TSV)
  ratings_path = "/Users/rezek_zhu/VLM/data/OASIS/ratings.tsv",

  # ── Metric-variant filter ──────────────────────────────────────────────────
  # List every metric__variant string you want to keep.
  # Format: "<metric>__<variant>"  (double underscore separator)
  metric_variants = c(
    "attention_map__100_mean_cls__mean",
    "attention_rollout__070_max_cls__mean",
    "cosine_similarity__semi"
  ),

  # ── Predictors ────────────────────────────────────────────────────────────
  # Each entry has three fields:
  #   col    – exact column name in the source data
  #   label  – human-readable label used in plots / tables
  #   source – "rating" (joined from ratings file) | "embedding" (from embedding cache)
  predictors = list(
    list(col = "Valence_mean", label = "Valence",    source = "rating"),
    list(col = "Arousal_mean", label = "Arousal",    source = "rating"),
    #list(col = "Arousal_SD",   label = "Arousal SD", source = "rating"),
    list(col = "emb_norm",     label = "Emb Norm",   source = "embedding")
  ),

  # ── Landscape plot axes ───────────────────────────────────────────────────
  # x / y must match a 'col' value in predictors above.
  landscape_valence = list(x = "Valence_mean", y = "Arousal_mean"),
  landscape_arousal = list(x = "Arousal_SD",   y = "Arousal_mean"),

  # ── Ratings filter (optional) ─────────────────────────────────────────────
  # list(col = "Category", value = "Scene")  →  keep only rows where Category == "Scene"
  # NULL  →  keep all rows
  ratings_filter = NULL,

  # ── Item-ID column in the ratings file ────────────────────────────────────
  ratings_id_col = "uniqueID",

  # ── Max scatter-plot points (down-sampled for speed) ──────────────────────
  scatter_max_n = 2000,

  # ── Module toggles ────────────────────────────────────────────────────────
  # Set a module to FALSE to skip it entirely; TRUE to run it.
  #
  #   cor_trial                – trial-level metric correlation heatmap
  #   cor_item                 – item-level metric correlation heatmap
  #   predictor_cor            – inter-predictor correlation heatmap
  #   item_regression          – item-level OLS regression, ALL predictors
  #                              (table + coef plot + scatter)
  #   item_regression_emotion  – item-level OLS regression, emotion (rating)
  #                              predictors only; no embedding
  #                              (table + coef plot + scatter)
  #   trial_mixed              – trial-level mixed model (table + coef plot + scatter)
  #   trial_regression_emotion – trial-level OLS regression, emotion difference
  #                              scores only; no embedding, no random effects
  #                              (table + coef plot + scatter)
  #   landscape_valence        – Valence x Arousal landscape scatter
  #   landscape_arousal        – Arousal SD x Arousal Mean landscape scatter
  run = list(
    cor_trial                = TRUE,
    cor_item                 = TRUE,
    predictor_cor            = TRUE,
    item_regression          = TRUE,
    item_regression_emotion  = TRUE,
    trial_mixed              = FALSE,
    trial_regression_emotion = TRUE,
    landscape_valence        = TRUE,
    landscape_arousal        = FALSE
  )
)
# =============================================================================
# END OF CONFIGURATION — do not edit below this line
# =============================================================================

# ── Libraries ─────────────────────────────────────────────────────────────────
library(tidyverse)
library(bruceR)
library(modelsummary)
library(broom)
library(parameters)
library(lme4)
library(lmerTest)
library(broom.mixed)
library(ggpubr)

# ── Derived settings ──────────────────────────────────────────────────────────
experiment_id <- basename(CFG$base_dir)
out_dir <- if (!is.null(CFG$out_dir)) CFG$out_dir else
  file.path(dirname(dirname(CFG$base_dir)), "results", experiment_id)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Convenience vectors derived from predictor config
pred_cols      <- sapply(CFG$predictors, `[[`, "col")
pred_labels    <- setNames(sapply(CFG$predictors, `[[`, "label"), pred_cols)
rating_cols    <- sapply(Filter(function(p) p$source == "rating",    CFG$predictors), `[[`, "col")
embedding_cols <- sapply(Filter(function(p) p$source == "embedding", CFG$predictors), `[[`, "col")

# Parse metric__variant strings into metric / variant columns for filtering
mv_parsed <- strsplit(CFG$metric_variants, "__", fixed = TRUE)
mv_filter_df <- data.frame(
  metric  = sapply(mv_parsed, `[[`, 1),
  variant = sapply(mv_parsed, function(x) paste(x[-1], collapse = "__")),
  stringsAsFactors = FALSE
)

# =============================================================================
# Data Prep  (always runs — all modules depend on these objects)
# =============================================================================
design_files     <- list.files(file.path(CFG$base_dir, "design_matrix"),
                               pattern = "\\.csv$", full.names = TRUE)
experiment_files <- list.files(file.path(CFG$base_dir, "experiment_data"),
                               pattern = "\\.csv$", full.names = TRUE)

df.design_matrix <- do.call(rbind, lapply(design_files, import))
df.experiment_data <- do.call(rbind, lapply(experiment_files, import)) %>%
  inner_join(mv_filter_df, by = c("metric", "variant")) %>%
  unite(metric_variant, metric, variant, sep = "__") %>%
  group_by(trial_id, metric_variant) %>%
  mutate(
    sum    = value1 + value2,
    value1 = value1 / sum,
    value2 = value2 / sum
  ) %>%
  ungroup() %>%
  select(-sum) %>%
  mutate(metric_variant = factor(metric_variant,
                                 levels = sort(CFG$metric_variants)))

df_trial <- inner_join(df.design_matrix, df.experiment_data, by = "trial_id")

df.embedding <- import(
  file.path(CFG$base_dir, "embedding_cache/item_embeddings_original.csv")) %>%
  rowwise() %>%
  mutate(emb_norm = sqrt(sum(c_across(starts_with("emb_dim"))^2))) %>%
  ungroup() %>%
  select(item_id, "emb_norm")

df.rating <- import(CFG$ratings_path)
if (!is.null(CFG$ratings_filter)) {
  df.rating <- df.rating %>%
    filter(.data[[CFG$ratings_filter$col]] == CFG$ratings_filter$value)
}

df_item_long <- df_trial %>%
  pivot_longer(
    cols          = c(item_1, item_2, value1, value2),
    names_to      = c(".value", "pos"),
    names_pattern = "(item|value)_?([12])"
  ) %>%
  select(-pos)

df_item <- df_item_long %>%
  group_by(item, metric_variant) %>%
  summarise(value = mean(value), .groups = "drop") %>%
  mutate(metric_variant = factor(metric_variant,
                                 levels = levels(df.experiment_data$metric_variant)))

# Joined item table — shared by regression, landscape, and predictor_cor modules
df_item_rating <- df_item %>%
  left_join(
    df.rating %>% select(all_of(c(CFG$ratings_id_col, rating_cols))),
    by = c("item" = CFG$ratings_id_col)
  ) %>%
  left_join(df.embedding, by = c("item" = "item_id")) %>%
  drop_na(all_of(pred_cols), value)

# =============================================================================
# Helper: Correlation heatmap
# =============================================================================
plot_cor_gg <- function(data, title_prefix = "Correlation") {
  cor_mat <- cor(data, use = "pairwise.complete.obs")
  n_obs   <- nrow(data)

  cor_df <- as.data.frame(as.table(cor_mat)) %>%
    as_tibble() %>%
    rename(var_x = Var1, var_y = Var2, r = Freq)

  ggplot(cor_df, aes(x = var_x, y = var_y, fill = r)) +
    geom_tile() +
    geom_text(aes(label = sprintf("%.2f", r)), size = 4) +
    scale_fill_gradient2(
      low = "#3B4CC0", mid = "white", high = "#B40426",
      midpoint = 0, limits = c(-1, 1)
    ) +
    coord_equal() +
    labs(title = paste0(title_prefix, " (N = ", n_obs, ")"),
         x = NULL, y = NULL, fill = "r") +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          panel.grid  = element_blank())
}

# =============================================================================
# Helper: Scatter plot  (facet: metric_variant x predictor)
# =============================================================================
plot_scatter <- function(data, x_vars, y_var,
                         x_labels = NULL, title = "", y_label = "Score",
                         max_n = CFG$scatter_max_n) {
  if (nrow(data) > max_n) data <- slice_sample(data, n = max_n)

  plot_df <- data %>%
    pivot_longer(cols = all_of(x_vars), names_to = "predictor", values_to = "x") %>%
    mutate(predictor = if (!is.null(x_labels))
      recode(predictor, !!!x_labels) else predictor)

  ggplot(plot_df, aes(x = x, y = .data[[y_var]])) +
    geom_point(alpha = 0.4, size = 0.8) +
    geom_smooth(method = "lm", se = TRUE, color = "#B40426") +
    stat_cor(method = "pearson", size = 2.5, label.sep = "\n") +
    facet_grid(metric_variant ~ predictor, scales = "free") +
    labs(title = title, x = "Predictor (raw)", y = y_label) +
    theme_bw() +
    theme(strip.text = element_text(size = 7))
}

# =============================================================================
# MODULE: Trial-level metric correlation heatmap
# =============================================================================
if (isTRUE(CFG$run$cor_trial)) {
  message("-- Running: cor_trial")

  df_trial_cor <- df_trial %>%
    select(-value2) %>%
    pivot_wider(names_from = metric_variant, values_from = value1) %>%
    select(-c(trial_id, item_1, item_2))

  p_cor_trial <- plot_cor_gg(df_trial_cor,
                             title_prefix = paste0("Trial-level correlation, N = ",
                                                   nrow(df_trial_cor)))
  print(p_cor_trial)
  ggsave(file.path(out_dir, "cor_trial_level.png"), p_cor_trial, width = 7, height = 6)
}

# =============================================================================
# MODULE: Item-level metric correlation heatmap
# =============================================================================
if (isTRUE(CFG$run$cor_item)) {
  message("-- Running: cor_item")

  df_item_cor <- df_item %>%
    pivot_wider(names_from = metric_variant, values_from = value) %>%
    select(-item)

  p_cor_item <- plot_cor_gg(df_item_cor,
                            title_prefix = paste0("Item-level correlation, N = ",
                                                  nrow(df_item_cor)))
  print(p_cor_item)
  ggsave(file.path(out_dir, "cor_item_level.png"), p_cor_item, width = 7, height = 6)
}

# =============================================================================
# MODULE: Item-level OLS regression — all predictors (incl. embedding)
# =============================================================================
if (isTRUE(CFG$run$item_regression)) {
  message("-- Running: item_regression")

  item_formula <- as.formula(
    paste("scale(value) ~",
          paste(sprintf("scale(%s)", pred_cols), collapse = " + "))
  )

  item_models <- df_item_rating %>%
    group_split(metric_variant) %>%
    set_names(levels(df_item_rating$metric_variant)) %>%
    map(~ lm(item_formula, data = .x))

  # Table
  item_rename <- setNames(
    paste0(pred_labels, " (std)"),
    sprintf("scale(%s)", pred_cols)
  )

  ms_item <- modelsummary(
    item_models,
    estimate    = "{estimate}{stars}",
    statistic   = NULL,
    coef_rename = item_rename,
    title       = "Item-level regression (standardized coefficients)",
    output      = "data.frame"
  )
  print(ms_item)
  write.csv(ms_item, file.path(out_dir, "table_item_level_regression.csv"),
            row.names = FALSE)

  # Coefficient plot
  item_coef_df <- imap_dfr(item_models, ~ {
    tidy(.x, conf.int = TRUE) %>%
      filter(term != "(Intercept)") %>%
      mutate(
        metric_variant = factor(.y, levels = levels(df_item_rating$metric_variant)),
        term = recode(term, !!!setNames(pred_labels, sprintf("scale(%s)", pred_cols)))
      )
  })

  p_item_coef <- ggplot(item_coef_df, aes(x = estimate, y = term)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
    geom_pointrange(aes(xmin = conf.low, xmax = conf.high)) +
    facet_wrap(~ metric_variant) +
    labs(title = "Item-level: standardized coefficients (95% CI)",
         x = "Standardized beta", y = NULL) +
    theme_bw()
  print(p_item_coef)
  ggsave(file.path(out_dir, "plot_item_coef.png"), p_item_coef, width = 9, height = 5)

  # Scatter plot
  p_item_scatter <- plot_scatter(
    data     = df_item_rating,
    x_vars   = pred_cols,
    y_var    = "value",
    x_labels = pred_labels,
    title    = "Item-level: outcome ~ each predictor",
    y_label  = "Mean attention score"
  )
  print(p_item_scatter)
  ggsave(file.path(out_dir, "plot_item_scatter.png"), p_item_scatter,
         width = 10, height = 6)
}

# =============================================================================
# MODULE: Item-level OLS regression — emotion predictors only (no embedding)
# =============================================================================
if (isTRUE(CFG$run$item_regression_emotion)) {
  message("-- Running: item_regression_emotion")

  item_emo_formula <- as.formula(
    paste("scale(value) ~",
          paste(sprintf("scale(%s)", rating_cols), collapse = " + "))
  )

  # Build item table using only rating predictors — more rows than df_item_rating
  # because embedding NAs no longer force exclusion
  df_item_rating_emo <- df_item %>%
    left_join(
      df.rating %>% select(all_of(c(CFG$ratings_id_col, rating_cols))),
      by = c("item" = CFG$ratings_id_col)
    ) %>%
    drop_na(all_of(rating_cols), value) %>%
    mutate(metric_variant = factor(metric_variant,
                                   levels = levels(df.experiment_data$metric_variant)))

  item_emo_models <- df_item_rating_emo %>%
    group_split(metric_variant) %>%
    set_names(levels(df_item_rating_emo$metric_variant)) %>%
    map(~ lm(item_emo_formula, data = .x))

  # Table
  item_emo_rename <- setNames(
    paste0(pred_labels[rating_cols], " (std)"),
    sprintf("scale(%s)", rating_cols)
  )

  ms_item_emo <- modelsummary(
    item_emo_models,
    estimate    = "{estimate}{stars}",
    statistic   = NULL,
    coef_rename = item_emo_rename,
    title       = "Item-level regression — emotion only (standardized coefficients)",
    output      = "data.frame"
  )
  print(ms_item_emo)
  write.csv(ms_item_emo,
            file.path(out_dir, "table_item_level_regression_emotion.csv"),
            row.names = FALSE)

  # Coefficient plot
  item_emo_coef_df <- imap_dfr(item_emo_models, ~ {
    tidy(.x, conf.int = TRUE) %>%
      filter(term != "(Intercept)") %>%
      mutate(
        metric_variant = factor(.y, levels = levels(df_item_rating_emo$metric_variant)),
        term = recode(term,
                      !!!setNames(pred_labels[rating_cols],
                                  sprintf("scale(%s)", rating_cols)))
      )
  })

  p_item_emo_coef <- ggplot(item_emo_coef_df, aes(x = estimate, y = term)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
    geom_pointrange(aes(xmin = conf.low, xmax = conf.high)) +
    facet_wrap(~ metric_variant) +
    labs(title = "Item-level (emotion only): standardized coefficients (95% CI)",
         x = "Standardized beta", y = NULL) +
    theme_bw()
  print(p_item_emo_coef)
  ggsave(file.path(out_dir, "plot_item_coef_emotion.png"),
         p_item_emo_coef, width = 9, height = 5)

  # Scatter plot
  p_item_emo_scatter <- plot_scatter(
    data     = df_item_rating_emo,
    x_vars   = rating_cols,
    y_var    = "value",
    x_labels = pred_labels[rating_cols],
    title    = "Item-level (emotion only): outcome ~ each predictor",
    y_label  = "Mean attention score"
  )
  print(p_item_emo_scatter)
  ggsave(file.path(out_dir, "plot_item_scatter_emotion.png"),
         p_item_emo_scatter, width = 10, height = 6)
}

# =============================================================================
# MODULE: Trial-level mixed model
# =============================================================================
if (isTRUE(CFG$run$trial_mixed)) {
  message("-- Running: trial_mixed")

  # Step 1: canonicalize pairs
  df_trial_mixed <- df_trial %>%
    mutate(
      item_A   = if_else(item_1 < item_2, item_1, item_2),
      item_Z   = if_else(item_1 < item_2, item_2, item_1),
      position = if_else(item_1 == item_A, 1L, 2L),
      value_A  = if_else(position == 1L, value1, 1 - value1),
      position = factor(position)
    )

  # Step 2: attach ratings + embeddings for A and Z
  ratings_sub <- df.rating %>% select(all_of(c(CFG$ratings_id_col, rating_cols)))
  emb_sub     <- df.embedding

  for (side in c("A", "Z")) {
    id_col <- paste0("item_", side)
    df_trial_mixed <- df_trial_mixed %>%
      left_join(ratings_sub, by = setNames(CFG$ratings_id_col, id_col)) %>%
      rename_with(~ paste0(.x, "_", side), all_of(rating_cols)) %>%
      left_join(emb_sub, by = setNames("item_id", id_col)) %>%
      rename_with(~ paste0(.x, "_", side), all_of(embedding_cols))
  }

  # Step 3: difference scores
  diff_cols <- paste0(pred_cols, "_diff")
  for (p in pred_cols) {
    df_trial_mixed[[paste0(p, "_diff")]] <-
      df_trial_mixed[[paste0(p, "_A")]] - df_trial_mixed[[paste0(p, "_Z")]]
  }
  df_trial_mixed <- df_trial_mixed %>% drop_na(all_of(diff_cols), value_A)

  # Step 4: mixed model formula (random effects fixed)
  trial_formula <- as.formula(
    paste(
      "scale(value_A) ~",
      paste(sprintf("scale(%s)", diff_cols), collapse = " + "),
      "+ position + (1 | item_A) + (1 | item_Z)"
    )
  )

  trial_mixed_models <- df_trial_mixed %>%
    group_split(metric_variant) %>%
    set_names(levels(df_trial_mixed$metric_variant)) %>%
    map(~ lmer(trial_formula, data = .x, REML = TRUE))

  # Step 5: table
  trial_rename <- c(
    setNames(paste0(pred_labels, " diff (std)"), sprintf("scale(%s_diff)", pred_cols)),
    "position2" = "Position (Z first)"
  )

  ms_trial <- modelsummary(
    trial_mixed_models,
    estimate    = "{estimate}{stars}",
    statistic   = NULL,
    coef_rename = trial_rename,
    title       = "Trial-level mixed model (standardized coefficients)",
    output      = "data.frame"
  )
  print(ms_trial)
  write.csv(ms_trial, file.path(out_dir, "table_trial_mixed_model.csv"),
            row.names = FALSE)

  # Step 6: coefficient plot
  trial_term_labels <- c(
    setNames(paste0(pred_labels, " diff"), sprintf("scale(%s_diff)", pred_cols)),
    "position2" = "Position (Z first)"
  )

  trial_mixed_coef_df <- imap_dfr(trial_mixed_models, ~ {
    broom.mixed::tidy(.x, effects = "fixed", conf.int = TRUE) %>%
      filter(term != "(Intercept)") %>%
      mutate(
        metric_variant = factor(.y, levels = levels(df_trial_mixed$metric_variant)),
        term = recode(term, !!!trial_term_labels)
      )
  })

  p_trial_coef <- ggplot(trial_mixed_coef_df, aes(x = estimate, y = term)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
    geom_pointrange(aes(xmin = conf.low, xmax = conf.high)) +
    facet_wrap(~ metric_variant) +
    labs(title = "Trial-level mixed model: fixed effects (95% CI)",
         x = "Standardized beta", y = NULL) +
    theme_bw()
  print(p_trial_coef)
  ggsave(file.path(out_dir, "plot_trial_coef.png"), p_trial_coef, width = 9, height = 5)

  # Step 7: scatter plot
  p_trial_scatter <- plot_scatter(
    data     = df_trial_mixed,
    x_vars   = diff_cols,
    y_var    = "value_A",
    x_labels = setNames(paste0(pred_labels, " diff"), diff_cols),
    title    = "Trial-level: outcome ~ each predictor",
    y_label  = "Item A attention share"
  )
  print(p_trial_scatter)
  ggsave(file.path(out_dir, "plot_trial_scatter.png"), p_trial_scatter,
         width = 10, height = 6)
}

# =============================================================================
# MODULE: Trial-level OLS regression — emotion predictors only (no embedding,
#         no random effects); uses within-pair emotion difference scores
# =============================================================================
if (isTRUE(CFG$run$trial_regression_emotion)) {
  message("-- Running: trial_regression_emotion")

  # Step 1: canonicalize pairs
  df_trial_emo <- df_trial %>%
    mutate(
      item_A   = if_else(item_1 < item_2, item_1, item_2),
      item_Z   = if_else(item_1 < item_2, item_2, item_1),
      position = if_else(item_1 == item_A, 1L, 2L),
      value_A  = if_else(position == 1L, value1, 1 - value1),
      position = factor(position)
    )

  # Step 2: attach emotion ratings only for A and Z
  ratings_sub <- df.rating %>% select(all_of(c(CFG$ratings_id_col, rating_cols)))

  for (side in c("A", "Z")) {
    id_col <- paste0("item_", side)
    df_trial_emo <- df_trial_emo %>%
      left_join(ratings_sub, by = setNames(CFG$ratings_id_col, id_col)) %>%
      rename_with(~ paste0(.x, "_", side), all_of(rating_cols))
  }

  # Step 3: emotion difference scores
  emo_diff_cols <- paste0(rating_cols, "_diff")
  for (p in rating_cols) {
    df_trial_emo[[paste0(p, "_diff")]] <-
      df_trial_emo[[paste0(p, "_A")]] - df_trial_emo[[paste0(p, "_Z")]]
  }
  df_trial_emo <- df_trial_emo %>%
    drop_na(all_of(emo_diff_cols), value_A) %>%
    mutate(metric_variant = factor(metric_variant,
                                   levels = levels(df.experiment_data$metric_variant)))

  # Step 4: OLS formula — emotion diffs + position, no random effects
  trial_emo_formula <- as.formula(
    paste(
      "scale(value_A) ~",
      paste(sprintf("scale(%s)", emo_diff_cols), collapse = " + "),
      "+ position"
    )
  )

  trial_emo_models <- df_trial_emo %>%
    group_split(metric_variant) %>%
    set_names(levels(df_trial_emo$metric_variant)) %>%
    map(~ lm(trial_emo_formula, data = .x))

  # Step 5: table
  trial_emo_rename <- c(
    setNames(
      paste0(pred_labels[rating_cols], " diff (std)"),
      sprintf("scale(%s_diff)", rating_cols)
    ),
    "position2" = "Position (Z first)"
  )

  ms_trial_emo <- modelsummary(
    trial_emo_models,
    estimate    = "{estimate}{stars}",
    statistic   = NULL,
    coef_rename = trial_emo_rename,
    title       = "Trial-level OLS regression — emotion only (standardized coefficients)",
    output      = "data.frame"
  )
  print(ms_trial_emo)
  write.csv(ms_trial_emo,
            file.path(out_dir, "table_trial_regression_emotion.csv"),
            row.names = FALSE)

  # Step 6: coefficient plot
  trial_emo_term_labels <- c(
    setNames(
      paste0(pred_labels[rating_cols], " diff"),
      sprintf("scale(%s_diff)", rating_cols)
    ),
    "position2" = "Position (Z first)"
  )

  trial_emo_coef_df <- imap_dfr(trial_emo_models, ~ {
    tidy(.x, conf.int = TRUE) %>%
      filter(term != "(Intercept)") %>%
      mutate(
        metric_variant = factor(.y, levels = levels(df_trial_emo$metric_variant)),
        term = recode(term, !!!trial_emo_term_labels)
      )
  })

  p_trial_emo_coef <- ggplot(trial_emo_coef_df, aes(x = estimate, y = term)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
    geom_pointrange(aes(xmin = conf.low, xmax = conf.high)) +
    facet_wrap(~ metric_variant) +
    labs(title = "Trial-level OLS (emotion only): standardized coefficients (95% CI)",
         x = "Standardized beta", y = NULL) +
    theme_bw()
  print(p_trial_emo_coef)
  ggsave(file.path(out_dir, "plot_trial_coef_emotion.png"),
         p_trial_emo_coef, width = 9, height = 5)

  # Step 7: scatter plot
  p_trial_emo_scatter <- plot_scatter(
    data     = df_trial_emo,
    x_vars   = emo_diff_cols,
    y_var    = "value_A",
    x_labels = setNames(paste0(pred_labels[rating_cols], " diff"), emo_diff_cols),
    title    = "Trial-level (emotion only): outcome ~ each predictor",
    y_label  = "Item A attention share"
  )
  print(p_trial_emo_scatter)
  ggsave(file.path(out_dir, "plot_trial_scatter_emotion.png"),
         p_trial_emo_scatter, width = 10, height = 6)
}

# =============================================================================
# MODULE: Valence landscape plot
# =============================================================================
if (isTRUE(CFG$run$landscape_valence)) {
  message("-- Running: landscape_valence")

  lv <- CFG$landscape_valence

  df_lv <- df_item_rating %>%
    group_by(metric_variant) %>%
    mutate(value_norm = scales::rescale(value, to = c(0, 1))) %>%
    ungroup()

  p_valence_landscape <- ggplot(df_lv,
                                aes(x = .data[[lv$x]], y = .data[[lv$y]],
                                    color = value_norm)) +
    geom_point(alpha = 0.8, size = 2) +
    facet_wrap(~ metric_variant) +
    scale_color_viridis_c() +
    labs(
      title = paste0("Item-level emotional landscape (",
                     pred_labels[lv$x], " x ", pred_labels[lv$y], ")"),
      x     = pred_labels[lv$x],
      y     = pred_labels[lv$y],
      color = "Attention (normalized)"
    ) +
    theme_bw()
  print(p_valence_landscape)
  ggsave(file.path(out_dir, "plot_item_landscape_valence.png"),
         p_valence_landscape, width = 10, height = 5)
}

# =============================================================================
# MODULE: Arousal landscape plot
# =============================================================================
if (isTRUE(CFG$run$landscape_arousal)) {
  message("-- Running: landscape_arousal")

  la <- CFG$landscape_arousal

  df_la <- df_item_rating %>%
    drop_na(all_of(la$x)) %>%
    group_by(metric_variant) %>%
    mutate(value_norm = scales::rescale(value, to = c(0, 1))) %>%
    ungroup()

  p_arousal_landscape <- ggplot(df_la,
                                aes(x = .data[[la$x]], y = .data[[la$y]],
                                    color = value_norm)) +
    geom_point(alpha = 0.8, size = 2) +
    facet_wrap(~ metric_variant) +
    scale_color_viridis_c() +
    labs(
      title = paste0("Item-level arousal landscape (",
                     pred_labels[la$x], " x ", pred_labels[la$y], ")"),
      x     = pred_labels[la$x],
      y     = pred_labels[la$y],
      color = "Attention (normalized)"
    ) +
    theme_bw()
  print(p_arousal_landscape)
  ggsave(file.path(out_dir, "plot_item_landscape_arousal.png"),
         p_arousal_landscape, width = 10, height = 5)
}

# =============================================================================
# MODULE: Predictor inter-correlation heatmap
# =============================================================================
if (isTRUE(CFG$run$predictor_cor)) {
  message("-- Running: predictor_cor")

  first_metric <- levels(df_item_rating$metric_variant)[1]

  df_corr  <- df_item_rating %>%
    filter(metric_variant == first_metric) %>%
    select(all_of(pred_cols))

  corr_mat <- cor(df_corr, use = "pairwise.complete.obs")
  colnames(corr_mat) <- rownames(corr_mat) <- pred_labels[colnames(corr_mat)]

  corr_long <- as.data.frame(as.table(corr_mat)) %>%
    as_tibble() %>%
    rename(var_x = Var1, var_y = Var2, r = Freq)

  p_pred_cor <- ggplot(corr_long, aes(x = var_x, y = var_y, fill = r)) +
    geom_tile() +
    geom_text(aes(label = sprintf("%.2f", r)), color = "black", size = 5) +
    scale_fill_gradient2(
      low = "#3B4CC0", mid = "white", high = "#B40426",
      midpoint = 0, limits = c(-1, 1)
    ) +
    coord_equal() +
    labs(title = "Correlation between predictors", x = NULL, y = NULL, fill = "r") +
    theme_bw() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      axis.text   = element_text(size = 12),
      plot.title  = element_text(size = 14, hjust = 0.5)
    )
  print(p_pred_cor)
  ggsave(file.path(out_dir, "plot_predictor_correlation.png"),
         p_pred_cor, width = 6, height = 5)
}

message("All results saved to: ", out_dir)
