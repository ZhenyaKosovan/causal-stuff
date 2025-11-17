\name{.recode_factor}
\alias{.recode_factor}
\title{Safe factorize id/time}
\usage{
.recode_factor(x)
}
\arguments{
\item{x}{Vector of ids or times to be converted into compact integer codes.}
}
\value{
List with \code{values} (integer encodings) and \code{levels} (the original factor levels).
}
\description{
Helper used internally to turn arbitrary ids/times into tightly packed integer codes
so that the C++ routines can operate on dense arrays without repeated factor work.
}
