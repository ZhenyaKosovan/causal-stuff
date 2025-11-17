test_that("sc weights are a simplex", {
  set.seed(1)
  X <- matrix(rnorm(50 * 6), 50, 6)
  y <- rnorm(6)
  w <- sc_weights(X, y, lambda = 1e-3, maxit = 2000)
  expect_true(all(w >= -1e-8))
  expect_true(abs(sum(w) - 1) < 1e-6)
})

test_that("sdid_fit matches synthdid defaults", {
  skip_if_not_installed("synthdid")
  data(california_prop99, package = "synthdid")
  pm <- synthdid::panel.matrices(california_prop99)
  Y <- pm[["Y"]]
  N0 <- pm[["N0"]]
  T0 <- pm[["T0"]]
  manual <- sdid_fit(
    Y_T_pre = Y[(N0 + 1):nrow(Y), 1:T0, drop = FALSE],
    Y_D_pre = Y[1:N0, 1:T0, drop = FALSE],
    Y_T_post = Y[(N0 + 1):nrow(Y), (T0 + 1):ncol(Y), drop = FALSE],
    Y_D_post = Y[1:N0, (T0 + 1):ncol(Y), drop = FALSE]
  )
  ref <- synthdid::synthdid_estimate(Y, N0, T0)
  weights <- attr(ref, "weights")
  expect_equal(as.numeric(manual$tau), as.numeric(ref), tolerance = 1e-6)
  expect_equal(as.numeric(manual$v), as.numeric(weights$omega), tolerance = 1e-6)
  expect_equal(as.numeric(manual$omega), as.numeric(weights$lambda), tolerance = 1e-6)
})

test_that("sdid_vcov_placebo matches synthdid", {
  skip_if_not_installed("synthdid")
  data(california_prop99, package = "synthdid")
  pm <- synthdid::panel.matrices(california_prop99)
  Y <- pm[["Y"]]
  N0 <- pm[["N0"]]
  T0 <- pm[["T0"]]
  set.seed(123)
  ours <- sdid_vcov_placebo(Y, N0, T0, replications = 50)
  set.seed(123)
  ref_est <- synthdid::synthdid_estimate(Y, N0, T0)
  ref <- vcov(ref_est, method = "placebo", replications = 50)
  expect_equal(as.numeric(ours), as.numeric(ref), tolerance = 1e-6)
})

test_that("sdid_fit filters incomplete pre columns", {
  Y_T_pre <- matrix(c(1, NA), nrow = 1)
  Y_D_pre <- matrix(c(1, 2), nrow = 1)
  Y_T_post <- matrix(3, nrow = 1)
  Y_D_post <- matrix(4, nrow = 1)
  expect_error(
    sdid_fit(Y_T_pre, Y_D_pre, Y_T_post, Y_D_post, lambda_unit = 0.1, lambda_time = 0.1),
    "fully observed"
  )
})

test_that("sdid_att validates inputs and runs on simple panel", {
  id <- rep(1:2, each = 3)
  t <- rep(1:3, times = 2)
  g <- c(2, 2, 2, NA, NA, NA)
  y <- c(1, 2, 3, 1, 2, 4)
  att <- sdid_att(id, t, y, g, L = 1, F = 1, min_cov = 1, lambda_unit = 0.1, lambda_time = 0.1)
  expect_named(att, c("att", "by_cohort"))
  expect_equal(nrow(att$by_cohort), 1)
})
