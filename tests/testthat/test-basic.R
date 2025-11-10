library(testthat)
library(panelsynth)


test_that("sc weights are a simplex", {
  set.seed(1)
  X <- matrix(rnorm(50 * 6), 50, 6)
  y <- rnorm(6)
  w <- sc_weights(X, y, lambda = 1e-3, maxit = 2000)
  expect_true(all(w >= -1e-8))
  expect_true(abs(sum(w) - 1) < 1e-6)
})
