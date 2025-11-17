\name{sdid_vcov_placebo}
\alias{sdid_vcov_placebo}
\title{Placebo variance for SDID estimates}
\usage{
sdid_vcov_placebo(
  Y,
  N0,
  T0,
  replications = 200L,
  lambda_unit = NULL,
  lambda_time = NULL,
  maxit = 10000L,
  tol = 1e-8
)
}
\arguments{
\item{Y}{Outcome matrix with donor units in the first \code{N0} rows and treated
units afterwards.}

\item{N0}{Number of donor units.}

\item{T0}{Number of pre-treatment periods.}

\item{replications}{Number of placebo samples.}

\item{lambda_unit}{Optional unit penalty to pass through to \code{\link{sdid_fit}}.}

\item{lambda_time}{Optional time penalty to pass through to \code{\link{sdid_fit}}.}

\item{maxit}{Maximum Frank-Wolfe iterations used by \code{\link{sdid_fit}}.}

\item{tol}{Relative convergence tolerance for \code{\link{sdid_fit}}.}
}
\value{
A \code{1 x 1} matrix containing the placebo variance estimate.
}
\description{
Computes the variance-covariance matrix of an SDID ATT using the placebo
permutation approach from \code{synthdid::vcov(method = "placebo")}.
}
