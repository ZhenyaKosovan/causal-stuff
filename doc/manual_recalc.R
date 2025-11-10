library(synthdid)
library(ggplot2)
set.seed(12345)


data("california_prop99")

ggplot(
  california_prop99 %>%
    mutate(treated = if_else(State == "California", TRUE, FALSE)),
  aes(x = Year, y = PacksPerCapita, color = treated, group = State)
) +
  geom_line(linewidth = 1.2) +
  geom_vline(aes(xintercept = 1988), color = "black", linetype = "dashed") +
  scale_color_manual(values = c("grey", "darkorange2")) +
  theme_minimal()
ggplot(
  california_prop99 %>%
    mutate(treated = if_else(State == "California", TRUE, FALSE)) %>%
    summarise(
      PacksPerCapita = mean(PacksPerCapita),
      .by = c(treated, Year)
    ) %>%
    filter(Year >= 1980),
  aes(x = Year, y = PacksPerCapita, color = treated)
) +
  geom_line(linewidth = 1.2) +
  geom_vline(aes(xintercept = 1988), color = "black", linetype = "dashed") +
  scale_color_manual(values = c("grey", "darkorange2")) +
  theme_minimal()


# 1. Omega estimation -----------------------------------------------------


# 1..1 Zeta ---------------------------------------------------------------
N_treated <- (dim(setup$Y)[[1]] - setup$N0)
T_pre <- colSums(setup$W)[colSums(setup$W) <= 0] %>% length()
T_post <- dim(setup$W)[[2]] - T_pre

calculate_sigma_hat <- function(Y, T_pre, N0) {
  # subset to N_co and T_pre - 1
  Y_sub <- Y[1:N0, 1:T_pre]
  # calculate Delta[i,t]
  Delta <- matrix(nrow = N0, ncol = T_pre - 1)
  rownames(Delta) <- rownames(Y_sub)
  colnames(Delta) <- colnames(Y_sub[1:T_pre - 1])
  # populate Delta matrix
  for (i in 1:N0) {
    for (j in 1:(T_pre - 1)) {
      Delta[i, j] <- Y_sub[i, j + 1] - Y_sub[i, j]
    }
  }
  Delta_hat <- 1 / (N0 * (T_pre - 1)) * sum(colSums(Delta))

  sigma_hat <- 1 / (N0 * (T_pre - 1)) * sum(colSums((Delta - Delta_hat)^2))

  sigma_hat
}

sigma_hat <- sigma_hat(setup$Y, T_pre, setup$N0)
zeta <- (N_treated * T_post)^0.25 * sigma_hat


# l-unit objective --------------------------------------------------------
l_unit <- function(Y, T_pre, N0, N_treated, omega, zeta) {
  l_unit <- omega[1]
  # Y[1:N0, 1:T_pre] is N0 x T_pre matrix, omega is N0 x 1 matrix
  # need to transpose before multiplication, then sum resulting T_pre x 1 vector
  # across the years
  l_unit <- l_unit + sum(t(Y[1:N0, 1:T_pre]) %*% as.matrix(omega[2:length(omega)]))
  # Subtract Treated units: weights are 1/N_treated (constant)
  # Subset matrix to only treated values and sum them up
  l_unit <- l_unit - 1 / N_treated * sum(Y[(N0 + 1):(N0 + N_treated), 1:T_pre])
  # Square the objective function
  l_unit <- l_unit^2
  # Add penalty
  l_unit <- l_unit + zeta^2 * T_pre * sqrt(sum(omega))
  l_unit
}

omega <- rep(1e-5, setup$N0 + 1)
l_unit(setup$Y, T_pre, setup$N0, N_treated, omega, zeta)

3333333
setup <- panel.matrices(california_prop99)
tau.hat <- synthdid_estimate(setup$Y, setup$N0, setup$T0)

tau.hat
#> synthdid: -15.604 +- NA. Effective N0/N0 = 16.4/38~0.4. Effective T0/T0 = 2.8/19~0.1. N1,T1 = 1,12.
print(summary(tau.hat))
#> $estimate
#> [1] -15.60383
#>
#> $se
#>      [,1]
#> [1,]   NA
#>
#> $controls
#>                estimate 1
#> Nevada              0.124
#> New Hampshire       0.105
#> Connecticut         0.078
#> Delaware            0.070
#> Colorado            0.058
#> Illinois            0.053
#> Nebraska            0.048
#> Montana             0.045
#> Utah                0.042
#> New Mexico          0.041
#> Minnesota           0.039
#> Wisconsin           0.037
#> West Virginia       0.034
#> North Carolina      0.033
#> Idaho               0.031
#> Ohio                0.031
#> Maine               0.028
#> Iowa                0.026
#>
#> $periods
#>      estimate 1
#> 1988      0.427
#> 1986      0.366
#> 1987      0.206
#>
#> $dimensions
#>           N1           N0 N0.effective           T1           T0 T0.effective
#>        1.000       38.000       16.388       12.000       19.000        2.783

se <- sqrt(vcov(tau.hat, method = "placebo"))
sprintf("point estimate: %1.2f", tau.hat)
#> [1] "point estimate: -15.60"
sprintf("95%% CI (%1.2f, %1.2f)", tau.hat - 1.96 * se, tau.hat + 1.96 * se)
#> [1] "95% CI (-32.01, 0.80)"
#>
plot(tau.hat, se.method = "placebo")
