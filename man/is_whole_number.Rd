\name{is_whole_number}
\alias{is_whole_number}
\title{Check whether a numeric vector is integer-valued}
\usage{
is_whole_number(x)
}
\arguments{
\item{x}{Numeric vector to test.}
}
\value{
Logical scalar; \code{TRUE} when every entry of \code{x} is within 1e-8 of an integer.
}
\description{
Convenience helper that verifies whether a numeric vector is effectively integer-valued.
}
\examples{
is_whole_number(1:5)
is_whole_number(c(1, 2.0000000001))
}
