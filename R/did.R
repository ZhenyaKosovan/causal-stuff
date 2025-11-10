#' Simple cohort/event-time DID ATT
#'
#' Computes ATT(g) for each first-treatment cohort g by comparing treated units
#' to controls (never-treated + not-yet-treated at g), within a pre/post window.
#'
#' @param id unit id vector
#' @param t time id vector (integer or coercible)
#' @param y outcome
#' @param g first-treatment time per unit (Inf/NA if never treated)
#' @param L pre periods count
#' @param F post periods count
#' @param min_cov coverage threshold in pre window (0..1)
#' @return list(att=overall ATT, by_cohort=data.frame(g,tau,n_treated))
#' @export
.did_core <- function(id, t, y, g, L, F, min_cov) {
  idr <- .recode_factor(id)
  tr <- .recode_factor(t)
  .did_cohort_att(
    as.integer(idr$values), as.integer(tr$values), as.numeric(y),
    as.integer(g), as.integer(L), as.integer(F), as.numeric(min_cov)
  )
}


#' @export
did_att <- function(id, t, y, g, L = 4L, F = 4L, min_cov = 0.8) {
  out <- .did_core(id, t, y, g, L, F, min_cov)
  out
}


#' (Optional) event-time aggregator placeholder
#' @export
did_event_agg <- function(...) stop("Not yet implemented in this prototype")
