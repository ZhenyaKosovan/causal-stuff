\name{sdid_att}
\alias{sdid_att}
\title{High-level SDID across cohorts in an unbalanced panel}
\usage{
sdid_att(
  id,
  t,
  y,
  g,
  L = 4L,
  F = 4L,
  min_cov = 0.8,
  lambda_unit = NULL,
  lambda_time = NULL,
  min_decrease = NULL,
  maxit = 10000L,
  tol = 1e-08
)
}
\arguments{
\item{id}{Unit identifiers.}

\item{t}{Time identifiers (numeric or coercible to integer).}

\item{y}{Outcome vector.}

\item{g}{Per-unit first-treatment time (\code{NA} for never treated).}

\item{L}{Number of pre-treatment periods to include in each cohort window.}

\item{F}{Number of post-treatment periods to include in each cohort window.}

\item{min_cov}{Minimum fraction of observed pre-period outcomes required to keep a unit.}

\item{lambda_unit}{Penalty (zeta) for donor weights. Defaults to the
\code{synthdid} heuristic when \code{NULL}.}

\item{lambda_time}{Penalty (zeta) for pre-period time weights. Defaults to
the \code{synthdid} heuristic when \code{NULL}.}

\item{min_decrease}{Absolute improvement tolerance shared by the underlying
Frank--Wolfe optimizers. Defaults to \code{1e-5 * noise}.}

\item{maxit}{Maximum Frank--Wolfe iterations.}

\item{tol}{Relative decrease tolerance.}
}
\value{
List with the aggregate ATT (\code{att}) and a data frame of cohort/event-time
estimates (\code{by_cohort}).
}
\description{
Convenience wrapper that repeatedly calls the SDID optimizer across treatment
cohorts within an unbalanced panel dataset.
}
