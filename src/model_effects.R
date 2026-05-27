fit_score_lm <- function(df) {
  df$model <- factor(df$model)
  df$comparison_set_id <- factor(df$comparison_set_id)
  df$entity_id <- factor(df$entity_id)
  df$assay_instance_hash <- factor(df$assay_instance_hash)

  contrasts(df$model) <- contr.sum(nlevels(df$model))
  contrasts(df$comparison_set_id) <- contr.sum(nlevels(df$comparison_set_id))

  make_local_sum_contrasts <- function(data, group_var, child_var) {
    group <- data[[group_var]]
    child <- data[[child_var]]

    out <- NULL

    for (g in levels(group)) {
      idx <- group == g
      kids <- sort(unique(as.character(child[idx])))

      if (length(kids) <= 1) next

      C <- contr.sum(length(kids))
      rownames(C) <- kids

      M <- matrix(
        0,
        nrow = nrow(data),
        ncol = ncol(C)
      )

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

      out <- cbind(out, M)
    }

    out
  }

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
    data = df
  )

  fit_summary <- summary(fit)

  beta <- coef(fit)
  V <- vcov(fit)
  coef_names <- names(beta)
  df_residual <- fit$df.residual

  f_statistic <- NA_real_
  f_numdf <- NA_real_
  f_dendf <- NA_real_
  f_p_value <- NA_real_

  if (!is.null(fit_summary$fstatistic)) {
    f_statistic <- unname(fit_summary$fstatistic[["value"]])
    f_numdf <- unname(fit_summary$fstatistic[["numdf"]])
    f_dendf <- unname(fit_summary$fstatistic[["dendf"]])
    f_p_value <- pf(
      f_statistic,
      f_numdf,
      f_dendf,
      lower.tail = FALSE
    )
  }

  regression_statistics <- data.frame(
    nobs = length(fitted(fit)),
    rank = fit$rank,
    df_residual = fit$df.residual,
    r_squared = fit_summary$r.squared,
    adj_r_squared = fit_summary$adj.r.squared,
    sigma = fit_summary$sigma,
    f_statistic = f_statistic,
    f_numdf = f_numdf,
    f_dendf = f_dendf,
    f_p_value = f_p_value,
    aic = AIC(fit),
    bic = BIC(fit)
  )

  empty_weights <- function() {
    setNames(numeric(0), character(0))
  }

  factor_weights <- function(prefix, levels_vec, level) {
    if (length(levels_vec) <= 1) {
      return(empty_weights())
    }

    C <- contr.sum(length(levels_vec))
    rownames(C) <- levels_vec

    w <- C[as.character(level), , drop = TRUE]
    names(w) <- paste0(prefix, seq_along(w))

    w
  }

  local_weights <- function(matrix_name, group_var, child_var, group_level, child_level) {
    group <- df[[group_var]]
    child <- df[[child_var]]

    idx <- as.character(group) == as.character(group_level)
    kids <- sort(unique(as.character(child[idx])))

    if (length(kids) <= 1) {
      return(empty_weights())
    }

    C <- contr.sum(length(kids))
    rownames(C) <- kids

    w <- C[as.character(child_level), , drop = TRUE]

    local_names <- paste0(
      child_var,
      "_within_",
      group_var,
      "[",
      group_level,
      "]_",
      seq_along(w)
    )

    names(w) <- paste0(matrix_name, local_names)

    w
  }

  interaction_weights <- function(left, right) {
    if (length(left) == 0 || length(right) == 0) {
      return(empty_weights())
    }

    values <- as.vector(outer(left, right))
    names(values) <- as.vector(outer(
      names(left),
      names(right),
      paste,
      sep = ":"
    ))

    values
  }

  estimate_term <- function(term, weights) {
    L <- setNames(rep(0, length(beta)), coef_names)

    for (name in names(weights)) {
      L[name] <- L[name] + weights[name]
    }

    active <- names(L)[abs(L) > 0]

    if (length(active) == 0) {
      return(data.frame(
        term = term,
        estimate = 0,
        std_error = 0,
        statistic = NA_real_,
        p_value = NA_real_,
        aliased = FALSE
      ))
    }

    aliased <- any(is.na(beta[active])) ||
      any(is.na(V[active, active, drop = FALSE]))

    if (aliased) {
      return(data.frame(
        term = term,
        estimate = NA_real_,
        std_error = NA_real_,
        statistic = NA_real_,
        p_value = NA_real_,
        aliased = TRUE
      ))
    }

    La <- L[active]
    estimate <- sum(La * beta[active])
    variance <- as.numeric(t(La) %*% V[active, active, drop = FALSE] %*% La)
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

    data.frame(
      term = term,
      estimate = estimate,
      std_error = std_error,
      statistic = statistic,
      p_value = p_value,
      aliased = FALSE
    )
  }

  observed <- function(cols) {
    out <- unique(as.data.frame(
      lapply(df[cols], as.character),
      stringsAsFactors = FALSE
    ))

    out[do.call(order, out), , drop = FALSE]
  }

  model_levels <- levels(df$model)
  comparison_set_levels <- levels(df$comparison_set_id)

  rows <- list()

  add <- function(term, weights) {
    rows[[length(rows) + 1]] <<- estimate_term(term, weights)
  }

  add(
    "(Intercept)",
    setNames(1, "(Intercept)")
  )

  for (model in model_levels) {
    add(
      paste0("model[", model, "]"),
      factor_weights("model", model_levels, model)
    )
  }

  for (comparison_set_id in comparison_set_levels) {
    add(
      paste0("comparison_set_id[", comparison_set_id, "]"),
      factor_weights(
        "comparison_set_id",
        comparison_set_levels,
        comparison_set_id
      )
    )
  }

  model_comparison_sets <- observed(c("model", "comparison_set_id"))

  for (i in seq_len(nrow(model_comparison_sets))) {
    model <- model_comparison_sets$model[[i]]
    comparison_set_id <- model_comparison_sets$comparison_set_id[[i]]

    add(
      paste0(
        "model[",
        model,
        "]:comparison_set_id[",
        comparison_set_id,
        "]"
      ),
      interaction_weights(
        factor_weights("model", model_levels, model),
        factor_weights(
          "comparison_set_id",
          comparison_set_levels,
          comparison_set_id
        )
      )
    )
  }

  comparison_set_entities <- observed(c("comparison_set_id", "entity_id"))

  for (i in seq_len(nrow(comparison_set_entities))) {
    comparison_set_id <- comparison_set_entities$comparison_set_id[[i]]
    entity_id <- comparison_set_entities$entity_id[[i]]

    add(
      paste0(
        "entity_id[",
        entity_id,
        "]|comparison_set_id[",
        comparison_set_id,
        "]"
      ),
      local_weights(
        "E_within_set",
        "comparison_set_id",
        "entity_id",
        comparison_set_id,
        entity_id
      )
    )
  }

  comparison_set_assay_instances <- observed(
    c("comparison_set_id", "assay_instance_hash")
  )

  for (i in seq_len(nrow(comparison_set_assay_instances))) {
    comparison_set_id <- comparison_set_assay_instances$comparison_set_id[[i]]
    assay_instance_hash <- comparison_set_assay_instances$assay_instance_hash[[i]]

    add(
      paste0(
        "assay_instance_hash[",
        assay_instance_hash,
        "]|comparison_set_id[",
        comparison_set_id,
        "]"
      ),
      local_weights(
        "A_within_set",
        "comparison_set_id",
        "assay_instance_hash",
        comparison_set_id,
        assay_instance_hash
      )
    )
  }

  model_comparison_set_entities <- observed(
    c("model", "comparison_set_id", "entity_id")
  )

  for (i in seq_len(nrow(model_comparison_set_entities))) {
    model <- model_comparison_set_entities$model[[i]]
    comparison_set_id <- model_comparison_set_entities$comparison_set_id[[i]]
    entity_id <- model_comparison_set_entities$entity_id[[i]]

    add(
      paste0(
        "model[",
        model,
        "]:entity_id[",
        entity_id,
        "]|comparison_set_id[",
        comparison_set_id,
        "]"
      ),
      interaction_weights(
        factor_weights("model", model_levels, model),
        local_weights(
          "E_within_set",
          "comparison_set_id",
          "entity_id",
          comparison_set_id,
          entity_id
        )
      )
    )
  }

  coefficients <- do.call(rbind, rows)
  rownames(coefficients) <- NULL

  list(
    coefficients = coefficients,
    regression_statistics = regression_statistics
  )
}