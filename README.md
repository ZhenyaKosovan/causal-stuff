# synthdid


```r
# install dev deps
# install.packages(c("Rcpp","RcppArmadillo","devtools","roxygen2","testthat","dplyr","tidyr"))


devtools::load_all(".") # after cloning this folder
Rcpp::compileAttributes()
devtools::document()


devtools::test()


# Synthetic control weights (no covariates)
set.seed(1)
X <- matrix(rnorm(80), 10, 8) # 10 donors x 8 pre periods
y <- rnorm(8)
w <- sc_weights(X, y, lambda=1e-3)


# With covariate balancing (e.g., GDP growth & CPI means over pre window)
Z <- matrix(rnorm(10*2), 10, 2) # donors x 2 covariates
z <- c(0.1, -0.2) # treated target covariates
w_cov <- sc_weights(X, y, Z=Z, z=z, cov_penalty=1e-2)


# Minimal SDID example (one cohort)
# sdid_fit(Y_T_pre, Y_D_pre, Y_T_post, Y_D_post, Z_D=Z, z_T=z, cov_penalty=1e-2)


# Panel-level SDID with covariates (R-level prototype)
# df must contain id, t, y, g, and covariate columns (e.g., gdp_growth, cpi)
# res <- sdid_att_cov(df, id=id, t=t, y=y, g=g, covars=c("gdp_growth","cpi"))
```
---


## Notes on parallelization & hotspots
- **OpenMP loops**: used to iterate cohorts independently in `did` and `sdid` (safe parallelism).
- **SC/SDID solvers**: inner PGD runs serially per weight vector; to parallelize further, parallelize across a **grid of lambdas** or across **treated units** (if you compute donor weights per unit instead of group-mean).
- **Memory**: Armadillo uses BLAS/LAPACK; link to vendor BLAS if you want even more speed.
- **CRAN-compat**: The Makevars use `$(SHLIB_OPENMP_CXXFLAGS)` which becomes empty if OpenMP isn’t present (no build failure).


---


## Where to extend
- Swap in exact SDID time-weight objective from the paper.
- Add **event-study** paths with cohort-specific ω (pre) and post extrapolation.
- Cross-validated penalties (`lambda_unit`, `lambda_time`) — parallelize across grid via OpenMP.
- Inference: moving block bootstrap by unit; placebo permutations for cohorts.
- More constraints: allow **sum-to-c ≤ 1** (subset SCM) and **weight sparsity** via entropic mirror descent.


---


## License
MIT
