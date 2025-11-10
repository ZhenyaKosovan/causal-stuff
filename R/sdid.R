#' Fit SDID weights and ATT for one cohort window
#'
#' @param Y_T_pre matrix (treated units x Tpre)
#' @param Y_D_pre matrix (donors x Tpre)
#' @param Y_T_post matrix (treated units x Tpost)
#' @param Y_D_post matrix (donors x Tpost)
#' @param lambda_unit ridge for unit weights
#' @param lambda_time ridge for time weights (smoothness/regularization)
#' @return list(v=unit weights, omega=time weights (length Tpre), tau=ATT)
#' @export
sdid_fit <- function(Y_T_pre, Y_D_pre, Y_T_post, Y_D_post,
                     lambda_unit = 1e-3, lambda_time = 1e-3,
                     maxit = 5000L, tol = 1e-8) {
  Y_T_pre <- as.matrix(Y_T_pre)
  Y_D_pre <- as.matrix(Y_D_pre)
  Y_T_post <- as.matrix(Y_T_post)
  Y_D_post <- as.matrix(Y_D_post)


  # target is treated mean pre path
  y_tgt <- colMeans(Y_T_pre, na.rm = TRUE)
  v <- .sc_pg_simplex(Y_D_pre, y_tgt, lambda_unit, as.integer(maxit), tol)


  # time weights: stabilize cross-sectional mean via smoothness objective on simplex
  mu <- colMeans(rbind(Y_T_pre, Y_D_pre), na.rm = TRUE)
  omega <- .time_weights_pg(mu, lambda_time, as.integer(maxit), tol)


  # helper to compute weighted averages ignoring NAs
  wmean_rows <- function(M, w) {
    apply(M, 1, function(row) {
      ok <- is.finite(row) & is.finite(w)
      if (!any(ok)) {
        return(NA_real_)
      }
      wloc <- w[ok] / sum(w[ok])
      sum(row[ok] * wloc)
    })
  }
  wmean_cols <- function(M, v) {
    ok <- is.finite(v) & rowSums(is.finite(M)) > 0
    vloc <- v[ok] / sum(v[ok])
    as.numeric(colSums(M[ok, , drop = FALSE] * vloc))
  }


  T_pre_hat_T <- mean(wmean_rows(Y_T_pre, omega), na.rm = TRUE)
  T_post_hat_T <- mean(wmean_rows(Y_T_post, omega), na.rm = TRUE)
  C_pre_hat <- mean(wmean_cols(Y_D_pre, v))
  C_post_hat <- mean(wmean_cols(Y_D_post, v))


  tau <- (T_post_hat_T - T_pre_hat_T) - (C_post_hat - C_pre_hat)
  list(v = v, omega = omega, tau = tau)
}


#' High-level SDID across cohorts in an unbalanced panel
#' @export
sdid_att <- function(id, t, y, g, L = 4L, F = 4L, min_cov = 0.8,
                     lambda_unit = 1e-3, lambda_time = 1e-3,
                     maxit = 5000L, tol = 1e-8) {
  idr <- .recode_factor(id)
  tr <- .recode_factor(t)
  .sdid_panel_att(
    as.integer(idr$values), as.integer(tr$values), as.numeric(y),
    as.integer(g), as.integer(L), as.integer(F), as.numeric(min_cov),
    lambda_unit, lambda_time, as.integer(maxit), tol
  )
}
