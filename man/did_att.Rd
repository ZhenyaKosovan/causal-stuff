\name{did_att}
\alias{did_att}
\title{Cohort-level difference-in-differences ATT}
\usage{
did_att(id, t, y, g, L = 4L, F = 4L, min_cov = 0.8)
}
\arguments{
\item{id}{Unit identifiers.}

\item{t}{Time identifiers (numeric or coercible to integer).}

\item{y}{Outcome vector.}

\item{g}{Per-unit first-treatment time (\code{NA} for never treated).}

\item{L}{Number of pre-treatment periods to include for each cohort.}

\item{F}{Number of post-treatment periods to include.}

\item{min_cov}{Minimum fraction of observed pre outcomes required for a unit.}
}
\value{
List containing the overall ATT (\code{att}) and a data frame with cohort-level
results (\code{by_cohort}).
}
\description{
Wrapper around the internal \code{.did_core()} helper that constructs classical
difference-in-differences estimates for each treatment cohort.
}
\examples{
set.seed(123)
id <- rep(1:5, each = 8)
t <- rep(2000:2007, times = 5)
g <- rep(NA_real_, length(id))
g[id \%in\% 4:5] <- 2004
y <- rnorm(length(id))
did_att(id, t, y, g, L = 2, F = 2)
}
