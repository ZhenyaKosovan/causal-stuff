#' Fit SDID weights and ATT for one cohort window
#'
#' @param Y_T_pre matrix (treated units x Tpre)
#' @param Y_D_pre matrix (donors x Tpre)
#' @param Y_T_post matrix (treated units x Tpost)
#' @param Y_D_post matrix (donors x Tpost)
#' @param lambda_unit penalty (zeta) for the unit-weight simplex problem. When
#'   `NULL`, the heuristic from `synthdid` is used.
#' @param lambda_time penalty (zeta) for the time-weight simplex problem. When
#'   `NULL`, the heuristic from `synthdid` is used.
#' @param maxit Maximum Frank-Wolfe iterations for both simplex optimizers.
#' @param tol Relative convergence tolerance.
#' @param min_decrease Optional absolute stopping tolerance for Frank-Wolfe
#'   improvements. When `NULL`, defaults to `1e-5 * noise` as in `synthdid`.
#' @param v_init Optional warm start for donor weights (length equal to donor
#'   units). Primarily used internally.
#' @param omega_init Optional warm start for time weights (length equals
#'   `T0`). Primarily used internally.
#' @return list(v=unit weights, omega=time weights (length Tpre), tau=ATT)
#' @export
sdid_fit <- function(Y_T_pre, Y_D_pre, Y_T_post, Y_D_post,
                     lambda_unit = NULL, lambda_time = NULL,
                     v_init = NULL, omega_init = NULL,
                     min_decrease = NULL,
                     maxit = 10000L, tol = 1e-8) {
  Y_T_pre <- as.matrix(Y_T_pre)
  Y_D_pre <- as.matrix(Y_D_pre)
  Y_T_post <- as.matrix(Y_T_post)
  Y_D_post <- as.matrix(Y_D_post)

  if (!is.numeric(Y_T_pre) || !is.numeric(Y_D_pre) ||
      !is.numeric(Y_T_post) || !is.numeric(Y_D_post)) {
    stop("All outcome inputs must be numeric matrices.")
  }
  if (any(!is.finite(Y_T_pre)) || any(!is.finite(Y_D_pre)))
    stop("Pre-period matrices must be fully observed (no NA/Inf).")
  if (ncol(Y_T_pre) != ncol(Y_D_pre))
    stop("Y_T_pre and Y_D_pre must have the same number of columns.")
  if (nrow(Y_T_pre) != nrow(Y_T_post))
    stop("Y_T_pre and Y_T_post must have the same number of rows (treated units).")
  if (nrow(Y_D_pre) != nrow(Y_D_post))
    stop("Y_D_pre and Y_D_post must have the same number of rows (donor units).")
  if (ncol(Y_T_post) != ncol(Y_D_post))
    stop("Y_T_post and Y_D_post must have the same number of columns (post periods).")
  if (ncol(Y_T_pre) == 0 || ncol(Y_T_post) == 0)
    stop("Require at least one pre- and one post-period column.")

  if (!is.null(lambda_unit) && (!is.numeric(lambda_unit) || any(lambda_unit < 0)))
    stop("lambda_unit must be non-negative.")
  if (!is.null(lambda_time) && (!is.numeric(lambda_time) || any(lambda_time < 0)))
    stop("lambda_time must be non-negative.")
  if (!is.null(min_decrease) && (!is.numeric(min_decrease) || any(min_decrease <= 0)))
    stop("min_decrease must be positive when provided.")
  if (maxit <= 0) stop("maxit must be positive.")
  if (tol <= 0) stop("tol must be positive.")

  if (nrow(Y_D_pre) == 0) stop("At least one donor unit is required.")
  if (nrow(Y_T_pre) == 0) stop("At least one treated unit is required.")

  penalties <- .sdid_penalties(Y_T_pre, Y_D_pre, Y_T_post, Y_D_post,
                               lambda_unit = lambda_unit,
                               lambda_time = lambda_time,
                               min_decrease = min_decrease)
  unit_warm <- if (is.null(v_init)) numeric() else as.numeric(v_init)
  if (length(unit_warm) && length(unit_warm) != nrow(Y_D_pre)) {
    warning("Ignoring v_init because its length does not match the number of donors.")
    unit_warm <- numeric()
  }
  time_warm <- if (is.null(omega_init)) numeric() else as.numeric(omega_init)
  if (length(time_warm)) {
    if (length(time_warm) != ncol(Y_D_pre)) {
      warning("Ignoring omega_init because its length does not match the number of pre periods.")
      time_warm <- numeric()
    }
  }

  res <- .sdid_fit_cohort(
    Y_T_pre = Y_T_pre,
    Y_D_pre = Y_D_pre,
    Y_T_post = Y_T_post,
    Y_D_post = Y_D_post,
    lambda_unit = penalties$lambda_unit,
    use_unit_default = FALSE,
    lambda_time = penalties$lambda_time,
    use_time_default = FALSE,
    maxit = as.integer(maxit),
    tol = tol,
    min_decrease = penalties$min_decrease,
    use_min_default = FALSE,
    unit_warm = unit_warm,
    time_warm = time_warm
  )
  list(v = res$v, omega = res$omega, tau = res$tau)
}


#' High-level SDID across cohorts in an unbalanced panel
#'
#' @inheritParams .did_core
#' @inheritParams sdid_fit
#' @return List with aggregate ATT and a data frame of cohort/event-time
#'   effects.
#' @export
sdid_att <- function(id, t, y, g, L = 4L, F = 4L, min_cov = 0.8,
                     lambda_unit = NULL, lambda_time = NULL,
                     min_decrease = NULL,
                     maxit = 10000L, tol = 1e-8) {
  if (length(id) != length(t) || length(id) != length(y) || length(id) != length(g))
    stop("id, t, y, and g must have the same length.")
  if (L <= 0 || F <= 0) stop("L and F must be positive.")
  if (!is.numeric(min_cov) || min_cov < 0 || min_cov > 1)
    stop("min_cov must be between 0 and 1.")
  idr <- .recode_factor(id)
  tr <- .recode_factor(t)
  g_levels <- match(g, tr$levels)

  N <- length(id)
  T_max <- length(tr$levels)
  I_max <- length(idr$levels)

  Y <- matrix(NA_real_, nrow = I_max, ncol = T_max)
  g_of <- rep(NA_integer_, I_max)
  for (k in seq_len(N)) {
    i <- idr$values[[k]]
    tt <- tr$values[[k]]
    Y[i, tt] <- as.numeric(y[[k]])
    if (!is.na(g_levels[[k]]) && is.na(g_of[[i]])) g_of[[i]] <- g_levels[[k]]
  }

  cohorts <- sort(unique(g_of[!is.na(g_of)]))
  if (length(cohorts) == 0) stop("No treated cohorts found (g contains only NA/Inf).")

  coverage <- function(unit, win) {
    inside <- win[win >= 1 & win <= T_max]
    if (length(inside) == 0) return(0)
    mean(is.finite(Y[unit, inside]))
  }

  tau_g <- rep(NA_real_, length(cohorts))
  n_treated <- integer(length(cohorts))
  dropped <- character()

  for (idx in seq_along(cohorts)) {
    gg <- cohorts[[idx]]
    pre_win <- seq.int(gg - L, gg - 1L)
    post_win <- seq.int(gg, gg + F - 1L)

    treated_ids <- which(g_of == gg)
    treated_ids <- treated_ids[sapply(treated_ids, coverage, win = pre_win) >= min_cov]
    donor_ids <- which(is.na(g_of) | g_of > gg)
    donor_ids <- donor_ids[sapply(donor_ids, coverage, win = pre_win) >= min_cov]

    if (length(treated_ids) == 0 || length(donor_ids) == 0) {
      dropped <- c(dropped, paste0("g=", tr$levels[[gg]], " (insufficient treated/donors with coverage)"))
      next
    }

    # assemble cohort matrices
    Tpre <- length(pre_win)
    Tpost <- length(post_win)
    Y_T_pre <- matrix(NA_real_, nrow = length(treated_ids), ncol = Tpre)
    Y_D_pre <- matrix(NA_real_, nrow = length(donor_ids), ncol = Tpre)
    Y_T_post <- matrix(NA_real_, nrow = length(treated_ids), ncol = Tpost)
    Y_D_post <- matrix(NA_real_, nrow = length(donor_ids), ncol = Tpost)

    for (r in seq_along(treated_ids)) {
      i <- treated_ids[[r]]
      for (c in seq_along(pre_win)) {
        tt <- pre_win[[c]]
        if (tt >= 1 && tt <= T_max) Y_T_pre[r, c] <- Y[i, tt]
      }
      for (c in seq_along(post_win)) {
        tt <- post_win[[c]]
        if (tt >= 1 && tt <= T_max) Y_T_post[r, c] <- Y[i, tt]
      }
    }
    for (r in seq_along(donor_ids)) {
      i <- donor_ids[[r]]
      for (c in seq_along(pre_win)) {
        tt <- pre_win[[c]]
        if (tt >= 1 && tt <= T_max) Y_D_pre[r, c] <- Y[i, tt]
      }
      for (c in seq_along(post_win)) {
        tt <- post_win[[c]]
        if (tt >= 1 && tt <= T_max) Y_D_post[r, c] <- Y[i, tt]
      }
    }

    fit <- tryCatch(
      sdid_fit(
        Y_T_pre = Y_T_pre,
        Y_D_pre = Y_D_pre,
        Y_T_post = Y_T_post,
        Y_D_post = Y_D_post,
        lambda_unit = lambda_unit,
        lambda_time = lambda_time,
        min_decrease = min_decrease,
        maxit = maxit,
        tol = tol
      ),
      error = function(e) {
        dropped <<- c(dropped, paste0("g=", tr$levels[[gg]], " (", e$message, ")"))
        NULL
      }
    )
    if (!is.null(fit)) {
      tau_g[[idx]] <- as.numeric(fit$tau)
      n_treated[[idx]] <- length(treated_ids)
    }
  }

  if (length(dropped))
    warning("Skipped cohorts: ", paste(dropped, collapse = "; "))

  num <- sum(tau_g * n_treated, na.rm = TRUE)
  den <- sum(n_treated[is.finite(tau_g)])
  att <- if (den > 0) num / den else NA_real_

  cohort_labels <- type.convert(as.character(tr$levels[cohorts]), as.is = TRUE)
  list(
    att = att,
    by_cohort = data.frame(
      g = cohort_labels,
      tau = tau_g,
      n_treated = n_treated,
      stringsAsFactors = FALSE
    )
  )
}


#' Placebo variance for SDID estimates
#'
#' Computes the variance-covariance matrix of an SDID ATT using placebo
#' re-assignments of treated units drawn from the donor pool. Matches the
#' `method = "placebo"` option from [`synthdid::vcov.synthdid_estimate()`].
#'
#' @param Y Outcome matrix with donor units in the first `N0` rows and treated
#'   units below.
#' @param N0 Number of donor units.
#' @param T0 Number of pre-treatment periods.
#' @param replications Number of placebo samples.
#' @inheritParams sdid_fit
#' @return `1 x 1` matrix containing the placebo variance estimate.
#' @export
sdid_vcov_placebo <- function(Y, N0, T0, replications = 200L,
                              lambda_unit = NULL, lambda_time = NULL,
                              maxit = 10000L, tol = 1e-8) {
  Y <- as.matrix(Y)
  N0 <- as.integer(N0)
  T0 <- as.integer(T0)
  replications <- as.integer(replications)

  if (is.na(N0) || N0 <= 0L || N0 >= nrow(Y))
    stop("N0 must be between 1 and nrow(Y) - 1")
  if (is.na(T0) || T0 <= 0L || T0 >= ncol(Y))
    stop("T0 must be between 1 and ncol(Y) - 1")
  if (is.na(replications) || replications < 2L)
    stop("replications must be at least 2")

  N1 <- nrow(Y) - N0
  if (N1 <= 0L) stop("Require at least one treated unit")
  if (N0 <= N1)
    stop("Placebo variance requires more donors than treated units")

  pre_cols <- seq_len(T0)
  post_cols <- seq.int(T0 + 1L, ncol(Y))

  donor_pre <- Y[seq_len(N0), pre_cols, drop = FALSE]
  noise_level <- if (ncol(donor_pre) < 2) 1 else stats::sd(as.vector(apply(donor_pre, 1, diff)))
  if (!is.finite(noise_level) || noise_level <= 0) noise_level <- 1
  eta <- max(1, as.numeric(N1) * length(post_cols))^(1/4)
  lambda_unit_val <- if (is.null(lambda_unit)) eta * noise_level else lambda_unit
  lambda_time_val <- if (is.null(lambda_time)) 1e-6 * noise_level else lambda_time
  min_decrease_val <- 1e-5 * noise_level

  base_fit <- sdid_fit(
    Y_T_pre = Y[(N0 + 1L):nrow(Y), pre_cols, drop = FALSE],
    Y_D_pre = Y[seq_len(N0), pre_cols, drop = FALSE],
    Y_T_post = Y[(N0 + 1L):nrow(Y), post_cols, drop = FALSE],
    Y_D_post = Y[seq_len(N0), post_cols, drop = FALSE],
    lambda_unit = lambda_unit_val,
    lambda_time = lambda_time_val,
    min_decrease = min_decrease_val,
    maxit = maxit,
    tol = tol
  )
  base_v <- as.numeric(base_fit$v)
  base_omega <- as.numeric(base_fit$omega)

  sum_normalize <- function(x) {
    s <- sum(x)
    if (s != 0) x / s else rep(1 / length(x), length(x))
  }

  placebo_tau <- function(idx) {
    n_donors <- length(idx) - N1
    donors <- idx[seq_len(n_donors)]
    treated <- idx[(n_donors + 1L):length(idx)]
    unit_init <- sum_normalize(base_v[donors])
    fit <- sdid_fit(
      Y_T_pre = Y[treated, pre_cols, drop = FALSE],
      Y_D_pre = Y[donors, pre_cols, drop = FALSE],
      Y_T_post = Y[treated, post_cols, drop = FALSE],
      Y_D_post = Y[donors, post_cols, drop = FALSE],
      lambda_unit = lambda_unit_val,
      lambda_time = lambda_time_val,
      v_init = unit_init,
      omega_init = base_omega,
      min_decrease = min_decrease_val,
      maxit = maxit,
      tol = tol
    )
    as.numeric(fit$tau)
  }

  taus <- replicate(replications, {
    idx <- sample.int(N0, size = N0, replace = FALSE)
    placebo_tau(idx)
  })

  var_hat <- ((replications - 1) / replications) * stats::var(taus)
  matrix(var_hat, nrow = 1L, ncol = 1L)
}
