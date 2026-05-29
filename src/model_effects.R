fit_score_lm <- function(df) {
  df$model <- factor(df$model)
  df$comparison_set_id <- factor(df$comparison_set_id)
  df$entity_id <- factor(df$entity_id)
  df$assay_instance_hash <- factor(df$assay_instance_hash)

  contrasts(df$model) <- contr.sum(nlevels(df$model))
  contrasts(df$comparison_set_id) <- contr.sum(nlevels(df$comparison_set_id))

  E_within_set <- make_local_sum_contrasts(
    df,
    group_var = "comparison_set_id",
    child_var = "entity_id"
  )

  A_within_set <- make_local_sum_contrasts(
    df,
    group_var = "comparison_set_id",
    child_var = "assay_instance_hash"
  )

  fit <- lm(
    score ~
      model * comparison_set_id +
      E_within_set +
      A_within_set +
      model:E_within_set,
    data = df,
    # tells R to store the design matrix it used; avoid reconstructing weights
    x = TRUE 
  )

  list(
    coefficients = term_contributions(
      fit,
      df,
      effect_keys_fn = score_effect_keys,
      effect_label_fn = score_effect_label
    ),
    regression_statistics = regression_statistics_for_fit(fit)
  )
}


fit_head_to_head_lpm <- function(df) {
  df$model <- factor(df$model)
  df$comparison_set_id <- factor(df$comparison_set_id)
  df$ordered_pair_id <- factor(df$ordered_pair_id)
  df$assay_instance_hash <- factor(df$assay_instance_hash)

  contrasts(df$model) <- contr.sum(nlevels(df$model))
  contrasts(df$comparison_set_id) <- contr.sum(nlevels(df$comparison_set_id))

  # one observation might be P>Q, and another might be Q>P.
  B_within_set <- make_local_sum_contrasts(
    df,
    group_var = "comparison_set_id",
    child_var = "ordered_pair_id"
  )

  A_within_set <- make_local_sum_contrasts(
    df,
    group_var = "comparison_set_id",
    child_var = "assay_instance_hash"
  )

  fit <- lm(
    score ~
      model * comparison_set_id +
      B_within_set +
      A_within_set +
      model:B_within_set,
    data = df,
    # tells R to store the design matrix it used; avoid reconstructing weights
    x = TRUE
  )

  list(
    coefficients = term_contributions(
      fit,
      df,
      effect_keys_fn = head_to_head_effect_keys,
      effect_label_fn = head_to_head_effect_label
    ),
    regression_statistics = regression_statistics_for_fit(fit)
  )
}


fit_steering_lm <- function(df) {
  df$model <- factor(df$model)
  df$comparison_set_id <- factor(df$comparison_set_id)
  df$directed_pair_id <- factor(df$directed_pair_id)
  df$assay_instance_hash <- factor(df$assay_instance_hash)

  contrasts(df$model) <- contr.sum(nlevels(df$model))
  contrasts(df$comparison_set_id) <- contr.sum(nlevels(df$comparison_set_id))

  D_within_set <- make_local_sum_contrasts(
    df,
    group_var = "comparison_set_id",
    child_var = "directed_pair_id"
  )

  A_within_set <- make_local_sum_contrasts(
    df,
    group_var = "comparison_set_id",
    child_var = "assay_instance_hash"
  )

  fit <- lm(
    score ~
      model * comparison_set_id +
      D_within_set +
      A_within_set +
      model:D_within_set,
    data = df,
    # tells R to store the design matrix it used; avoid reconstructing weights
    x = TRUE
  )

  list(
    coefficients = term_contributions(
      fit,
      df,
      effect_keys_fn = steering_effect_keys,
      effect_label_fn = steering_effect_label
    ),
    regression_statistics = regression_statistics_for_fit(fit)
  )
}


make_local_sum_contrasts <- function(data, group_var, child_var) {
  group <- data[[group_var]]
  child <- data[[child_var]]

  out <- matrix(numeric(0), nrow = nrow(data), ncol = 0)

  for (g in levels(group)) {
    idx <- group == g
    kids <- sort(unique(as.character(child[idx])))

    if (length(kids) <= 1) {
      next
    }

    C <- contr.sum(length(kids))
    rownames(C) <- kids # assigns levels/category members to rows in C

    M <- matrix(0, nrow = nrow(data), ncol = ncol(C)) # creates zero matrix
    # rows outside the current group get zeros, then the locally coded rows
    # are inserted
    M[idx, ] <- C[as.character(child[idx]), , drop = FALSE]

    colnames(M) <- paste0(
      child_var,
      "_within_",
      group_var,
      "[",
      g,
      "]_",
      seq_len(ncol(C))
    )

    out <- cbind(out, M) # concatenate horizontally to add next column
  }

  # design submatrix for the current locally constrained term
  # num rows = num observatinos
  out
}


regression_statistics_for_fit <- function(fit) {
  s <- summary(fit)

  f <- if (is.null(s$fstatistic)) {
    c(value = NA_real_, numdf = NA_real_, dendf = NA_real_)
  } else {
    s$fstatistic
  }

  data.frame(
    nobs = length(fitted(fit)),
    rank = fit$rank,
    df_residual = fit$df.residual,
    r_squared = s$r.squared,
    adj_r_squared = s$adj.r.squared,
    sigma = s$sigma,
    f_statistic = unname(f[["value"]]),
    f_numdf = unname(f[["numdf"]]),
    f_dendf = unname(f[["dendf"]]),
    f_p_value = if (is.na(f[["value"]])) {
      NA_real_
    } else {
      pf(f[["value"]], f[["numdf"]], f[["dendf"]], lower.tail = FALSE)
    },
    aic = AIC(fit),
    bic = BIC(fit)
  )
}


# creates row index lookup of design matrix to later retrieve
# the design weights for each individual combination of cols
representative_effect_rows <- function(df, cols) {
  if (length(cols) == 0) {
    return(data.frame(.row = 1L))
  }

  # key = [
  #     "\r".join(values)
  #     for values in zip(*(df[c].astype(str) for c in cols))
  # ]
  # gets all naturally occurring combinations of cols
  key <- do.call(
    paste,
    c(lapply(df[cols], as.character), sep = "\r")
  )

  keep <- !duplicated(key)

  # drop duplicates and stringify cols
  out <- df[keep, cols, drop = FALSE]
  out[] <- lapply(out, as.character)
  # add a .row col using rownumbers from original df for later lookup
  out$.row <- which(keep)

  out <- out[do.call(order, out[cols]), , drop = FALSE]
  # reindex rows ordinally (1, 2, 3, ...), does not impact .rows
  rownames(out) <- NULL

  out
}


# for each design col, maps it to its original regression term
# e.g. model1 or model2 -> model
term_labels_for_design <- function(fit) {
  labels <- c(
    "(Intercept)",
    attr(fit$terms, "term.labels")
  )

  labels[attr(fit$x, "assign") + 1L]
}


# maps model terms to original df columns
score_effect_keys <- function(model_term) {
  switch(
    model_term,
    "(Intercept)" = character(0),
    "model" = "model",
    "comparison_set_id" = "comparison_set_id",
    "model:comparison_set_id" = c("model", "comparison_set_id"),
    "E_within_set" = c("comparison_set_id", "entity_id"),
    "A_within_set" = c("comparison_set_id", "assay_instance_hash"),
    "model:E_within_set" = c("model", "comparison_set_id", "entity_id"),
    stop("Unknown model term: ", model_term)
  )
}


# maps model terms to original df columns
head_to_head_effect_keys <- function(model_term) {
  switch(
    model_term,
    "(Intercept)" = character(0),
    "model" = "model",
    "comparison_set_id" = "comparison_set_id",
    "model:comparison_set_id" = c("model", "comparison_set_id"),
    "B_within_set" = c("comparison_set_id", "ordered_pair_id"),
    "A_within_set" = c("comparison_set_id", "assay_instance_hash"),
    "model:B_within_set" = c("model", "comparison_set_id", "ordered_pair_id"),
    stop("Unknown model term: ", model_term)
  )
}


# maps model terms to original df columns
steering_effect_keys <- function(model_term) {
  switch(
    model_term,
    "(Intercept)" = character(0),
    "model" = "model",
    "comparison_set_id" = "comparison_set_id",
    "model:comparison_set_id" = c("model", "comparison_set_id"),
    "D_within_set" = c("comparison_set_id", "directed_pair_id"),
    "A_within_set" = c("comparison_set_id", "assay_instance_hash"),
    "model:D_within_set" = c("model", "comparison_set_id", "directed_pair_id"),
    stop("Unknown model term: ", model_term)
  )
}


# construct effect labels per calling Python expectation
score_effect_label <- function(model_term, row) {
  switch(
    model_term,
    "(Intercept)" = "(Intercept)",

    "model" = paste0(
      "model[",
      row$model,
      "]"
    ),

    "comparison_set_id" = paste0(
      "comparison_set_id[",
      row$comparison_set_id,
      "]"
    ),

    "model:comparison_set_id" = paste0(
      "model[",
      row$model,
      "]:comparison_set_id[",
      row$comparison_set_id,
      "]"
    ),

    "E_within_set" = paste0(
      "entity_id[",
      row$entity_id,
      "]|comparison_set_id[",
      row$comparison_set_id,
      "]"
    ),

    "A_within_set" = paste0(
      "assay_instance_hash[",
      row$assay_instance_hash,
      "]|comparison_set_id[",
      row$comparison_set_id,
      "]"
    ),

    "model:E_within_set" = paste0(
      "model[",
      row$model,
      "]:entity_id[",
      row$entity_id,
      "]|comparison_set_id[",
      row$comparison_set_id,
      "]"
    ),

    stop("Unknown model term: ", model_term)
  )
}


# construct effect labels per calling Python expectation
head_to_head_effect_label <- function(model_term, row) {
  switch(
    model_term,
    "(Intercept)" = "(Intercept)",

    "model" = paste0(
      "model[",
      row$model,
      "]"
    ),

    "comparison_set_id" = paste0(
      "comparison_set_id[",
      row$comparison_set_id,
      "]"
    ),

    "model:comparison_set_id" = paste0(
      "model[",
      row$model,
      "]:comparison_set_id[",
      row$comparison_set_id,
      "]"
    ),

    "B_within_set" = paste0(
      "beats[",
      row$ordered_pair_id,
      "]|comparison_set_id[",
      row$comparison_set_id,
      "]"
    ),

    "A_within_set" = paste0(
      "assay_instance_hash[",
      row$assay_instance_hash,
      "]|comparison_set_id[",
      row$comparison_set_id,
      "]"
    ),

    "model:B_within_set" = paste0(
      "model[",
      row$model,
      "]:beats[",
      row$ordered_pair_id,
      "]|comparison_set_id[",
      row$comparison_set_id,
      "]"
    ),

    stop("Unknown model term: ", model_term)
  )
}


# construct effect labels per calling Python expectation
steering_effect_label <- function(model_term, row) {
  switch(
    model_term,
    "(Intercept)" = "(Intercept)",

    "model" = paste0(
      "model[",
      row$model,
      "]"
    ),

    "comparison_set_id" = paste0(
      "comparison_set_id[",
      row$comparison_set_id,
      "]"
    ),

    "model:comparison_set_id" = paste0(
      "model[",
      row$model,
      "]:comparison_set_id[",
      row$comparison_set_id,
      "]"
    ),

    "D_within_set" = paste0(
      "steered[",
      row$directed_pair_id,
      "]|comparison_set_id[",
      row$comparison_set_id,
      "]"
    ),

    "A_within_set" = paste0(
      "assay_instance_hash[",
      row$assay_instance_hash,
      "]|comparison_set_id[",
      row$comparison_set_id,
      "]"
    ),

    "model:D_within_set" = paste0(
      "model[",
      row$model,
      "]:steered[",
      row$directed_pair_id,
      "]|comparison_set_id[",
      row$comparison_set_id,
      "]"
    ),

    stop("Unknown model term: ", model_term)
  )
}


effect_row <- function(
  term,
  estimate,
  std_error,
  statistic = NA_real_,
  p_value = NA_real_,
  aliased = FALSE
) {
  data.frame(
    term = term,
    estimate = estimate,
    std_error = std_error,
    statistic = statistic,
    p_value = p_value,
    aliased = aliased
  )
}


linear_effect <- function(
  beta,
  V,
  df_residual,
  term,
  weights
) {
  # local contrasts produce many zeros outside their own comparison set
  active <- names(weights)[abs(weights) > 0]

  if (length(active) == 0) {
    return(effect_row(term, 0, 0))
  }

  weights <- weights[active]

  # if a coefficient cannot be estimated cleanly, it will be given NA
  aliased <- any(is.na(beta[active])) ||
    any(is.na(V[active, active, drop = FALSE]))

  if (aliased) {
    return(effect_row(
      term = term,
      estimate = NA_real_,
      std_error = NA_real_,
      aliased = TRUE
    ))
  }

  # this will also produce the coefficient estimates of implied levels
  # since weights includes the implied [-1]*k weight
  estimate <- sum(weights * beta[active])

  # variance = w' V w
  variance <- as.numeric(
    t(weights) %*% V[active, active, drop = FALSE] %*% weights
  )

  std_error <- sqrt(max(variance, 0))

  statistic <- if (std_error == 0) {
    NA_real_
  } else {
    estimate / std_error
  }

  p_value <- if (is.na(statistic)) {
    NA_real_
  } else {
    2 * pt(abs(statistic), df = df_residual, lower.tail = FALSE)
  }

  effect_row(
    term = term,
    estimate = estimate,
    std_error = std_error,
    statistic = statistic,
    p_value = p_value
  )
}


# infers conceptual regression terms and computes their stats
term_contributions <- function(
  fit,
  df,
  effect_keys_fn,
  effect_label_fn
) {
  X <- fit$x # X is the design matrix used by the model
  design_terms <- term_labels_for_design(fit)

  beta <- coef(fit)
  V <- vcov(fit)

  # loop through every formula term
  rows <- lapply(unique(design_terms), function(model_term) {
    # we use df not X since we want each combination of original
    # columns for the current term not design columns (recall design
    # cols drop implied levels)
    lookup_rows <- representative_effect_rows(df, effect_keys_fn(model_term))
    term_cols <- design_terms == model_term

    rows_for_term <- lapply(seq_len(nrow(lookup_rows)), function(i) {
      row <- lookup_rows[i, , drop = FALSE]

      # lookup the row in the design matrix, keeping only cols belonging to
      # this specific formula term
      weights <- setNames(
        as.numeric(X[row$.row, term_cols, drop = TRUE]),
        colnames(X)[term_cols]
      )

      linear_effect(
        beta = beta,
        V = V,
        df_residual = fit$df.residual,
        term = effect_label_fn(model_term, row),
        weights = weights
      )
    })

    do.call(rbind, rows_for_term)
  })

  coefficients <- do.call(rbind, rows)
  rownames(coefficients) <- NULL

  coefficients
}