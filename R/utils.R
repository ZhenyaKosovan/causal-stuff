#' Check integerish
is_whole_number <- function(x) is.numeric(x) && all(abs(x - round(x)) < 1e-8)


#' Safe factorize id/time
.recode_factor <- function(x) {
  f <- as.integer(factor(x))
  list(values = f, levels = levels(factor(x)))
}
