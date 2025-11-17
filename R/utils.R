#' Check whether a numeric vector is integer-valued
#'
#' @param x Numeric vector.
#' @return `TRUE` when `x` is numeric and every entry is within `1e-8` of an integer.
#' @examples
#' is_whole_number(1:5)
#' is_whole_number(c(1, 2.0000000001))
is_whole_number <- function(x) is.numeric(x) && all(abs(x - round(x)) < 1e-8)


#' Safe factorize id/time
#'
#' Returns the integer codes along with the factor levels so C++ routines can
#' work with tightly packed identifiers.
#'
#' @param x Vector of ids/times.
#' @return List with `values` (integer codes) and `levels`.
#' @keywords internal
.recode_factor <- function(x) {
  f <- as.integer(factor(x))
  list(values = f, levels = levels(factor(x)))
}

#' Default SDID tuning parameters
#'
#' Implements the same heuristic choices used in the C++ routines and in
#' `synthdid`: zeta_unit scales with the donor noise and the size of the treated
#' x post window; zeta_time is a tiny ridge term; min_decrease scales with the
#' donor noise.
#' @keywords internal
.sdid_penalties <- function(Y_T_pre, Y_D_pre, Y_T_post, Y_D_post,
                           lambda_unit = NULL, lambda_time = NULL,
                           min_decrease = NULL) {
  N1 <- nrow(Y_T_pre)
  Tpost <- ncol(Y_D_post)

  noise <- .sdid_donor_noise_sd(Y_D_pre)
  eta <- max(1, as.numeric(N1) * as.numeric(Tpost))^(1 / 4)

  list(
    lambda_unit = if (is.null(lambda_unit)) eta * noise else lambda_unit,
    lambda_time = if (is.null(lambda_time)) 1e-6 * noise else lambda_time,
    min_decrease = if (is.null(min_decrease)) 1e-5 * noise else min_decrease
  )
}

#' Heuristic noise estimate used for defaults
#' @keywords internal
.sdid_donor_noise_sd <- function(donors) {
  if (!is.matrix(donors) || nrow(donors) == 0 || ncol(donors) < 2) return(1)
  diffs <- c()
  for (r in seq_len(nrow(donors))) {
    row_vals <- donors[r, ]
    if (sum(is.finite(row_vals)) < 2) next
    for (c in seq.int(2, ncol(donors))) {
      prev <- row_vals[[c - 1]]
      cur <- row_vals[[c]]
      if (is.finite(prev) && is.finite(cur)) diffs <- c(diffs, cur - prev)
    }
  }
  if (length(diffs) < 2) return(1)
  out <- stats::sd(diffs)
  if (!is.finite(out) || out <= 0) 1 else out
}
