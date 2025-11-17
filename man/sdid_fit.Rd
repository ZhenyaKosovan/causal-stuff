\name{sdid_fit}
\alias{sdid_fit}
\title{Fit SDID weights and ATT for one cohort window}
\usage{
sdid_fit(
  Y_T_pre,
  Y_D_pre,
  Y_T_post,
  Y_D_post,
  lambda_unit = NULL,
  lambda_time = NULL,
  v_init = NULL,
  omega_init = NULL,
  min_decrease = NULL,
  maxit = 10000L,
  tol = 1e-08
)
}
\arguments{
\item{Y_T_pre}{Matrix of treated units over the pre-treatment periods.}

\item{Y_D_pre}{Matrix of donor units over the pre-treatment periods.}

\item{Y_T_post}{Matrix of treated units over the post-treatment periods.}

\item{Y_D_post}{Matrix of donor units over the post-treatment periods.}

\item{lambda_unit}{Penalty (zeta) for the donor-weight simplex. When
\code{NULL}, the heuristic from the \code{synthdid} package is used.}

\item{lambda_time}{Penalty (zeta) for the pre-period time-weight simplex. When
\code{NULL}, the heuristic from the \code{synthdid} package is used.}

\item{v_init}{Optional donor-weight warm start used to initialize the Frank--Wolfe routine.}

\item{omega_init}{Optional time-weight warm start.}

\item{min_decrease}{Absolute improvement tolerance that terminates the
Frank--Wolfe steps. Defaults to \code{1e-5 * noise} to match
\code{synthdid}.}

\item{maxit}{Maximum Frank--Wolfe iterations shared by both simplex solvers.}

\item{tol}{Relative decrease tolerance that terminates the optimizer.}
}
\value{
List with donor weights \code{v}, time weights \code{omega}, and the SDID ATT \code{tau}.
}
\description{
Low-level optimizer that reproduces the synthetic difference-in-differences
solution for one treated-vs-donor cohort window.
}
