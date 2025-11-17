#' Synthetic Control donor weights (projected gradient on simplex)
#'
#' Supports optional **covariate balancing**: penalize mismatch between
#' donor-weighted covariates Z %*% w and treated covariate target `z`.
#'
#' Objective: 0.5||Xw - y||^2 + 0.5*cov_penalty||Zw - z||^2 + 0.5*lambda||w||^2,
#' subject to w >= 0, sum(w) = 1.
#'
#' @param X matrix: donors-by-T (pre periods)
#' @param y vector length T: treated pre path (or treated-group average)
#' @param Z optional matrix: donors-by-P covariates (e.g., **pre-window means** of GDP growth, CPI, ...)
#' @param z optional vector length P: treated-group covariate target (same construction as Z)
#' @param cov_penalty nonnegative scalar controlling the strength of covariate balancing
#' @param lambda ridge penalty (>=0)
#' @param maxit iterations
#' @param tol tolerance for relative improvement
#' @return numeric vector of donor weights (nonnegative, sum to 1)
#' @export
sc_weights <- function(X, y, Z = NULL, z = NULL, cov_penalty = 0,
                       lambda = 1e-3, maxit = 5000L, tol = 1e-8) {
  X <- as.matrix(X)
  y <- as.numeric(y)
  if (ncol(X) != length(y)) stop("X and y length mismatch")
  use_cov <- !is.null(Z) && !is.null(z) && cov_penalty > 0
  if (use_cov) {
    Z <- as.matrix(Z)
    z <- as.numeric(z)
    if (nrow(Z) != nrow(X)) stop("Z must have same number of rows as X (donors)")
    if (ncol(Z) != length(z)) stop("z must have length ncol(Z)")
    .sc_pg_simplex_cov(X, y, Z, z, as.numeric(cov_penalty), lambda, as.integer(maxit), tol)
  } else {
    .sc_pg_simplex(X, y, lambda, as.integer(maxit), tol)
  }
}
