// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
#include <RcppArmadillo.h>
#include "utils.h"

/*
 * sc.cpp
 * ------
 * Implements the optimization routines that compute synthetic control donor
 * weights. Both functions use projected gradient descent where the projection
 * step enforces that the weights live on the probability simplex. The
 * gradients are cheap to evaluate because the loss functions are quadratic.
 */

// -----------------------------------------------------------------------------
// sc_pg_simplex
//   X : donor-by-time matrix (each row is a donor unit, each column a period)
//   y : target time series for the treated unit
//   lambda : ridge penalty to stabilise the solution
//   maxit / tol : iteration controls for projected gradient descent
//
// The objective is 0.5||Xw - y||^2 + 0.5*lambda||w||^2 subject to w >= 0 and
// sum(w) = 1. The Armadillo SVD gives us a Lipschitz constant so we can pick
// a conservative step size without doing any line search.
// -----------------------------------------------------------------------------

// [[Rcpp::export(name = ".sc_pg_simplex")]]
Rcpp::NumericVector sc_pg_simplex(const arma::mat &X,
                                  const arma::vec &y,
                                  const double lambda = 1e-3,
                                  const int maxit = 5000,
                                  const double tol = 1e-8) {
  const arma::uword n = X.n_rows; // donors
  const arma::uword T = X.n_cols; // time
  if (y.n_elem != T) Rcpp::stop("y length mismatch with X columns");

  // Start from the uniform distribution; this satisfies all constraints and
  // represents the "no information" prior.
  arma::vec w(n, arma::fill::ones);
  w /= static_cast<double>(n);

  // Lipschitz bound via largest singular value. The gradient of the quadratic
  // loss is Lipschitz with constant sigma_max(X)^2 + lambda. Using 1/L keeps
  // the projected gradient steps stable without any tuning by the caller.
  arma::vec s;
  arma::mat U, V;
  arma::svd_econ(U, s, V, X, 'l');
  double L = (s.n_elem > 0 ? s.max() * s.max() : 1.0) + lambda;
  double step = 1.0 / L;

  double f_old = arma::datum::inf;

  for (int it = 0; it < maxit; ++it) {
    arma::vec Xw = X * w;                        // fitted path under current w
    arma::vec grad = X.t() * (Xw - y) + lambda * w; // gradient of objective

    // Unconstrained descent step, then projection brings us back to simplex.
    arma::vec w_new = project_simplex(w - step * grad);

    // Evaluate new objective to check for relative progress.
    double f_new = 0.5 * arma::dot(Xw - y, Xw - y) + 0.5 * lambda * arma::dot(w, w);

    if (rel_improve(f_old, f_new) < tol) {
      w = w_new;
      break;
    }
    w = w_new;
    f_old = f_new;
  }

  return Rcpp::wrap(Rcpp::NumericVector(w.begin(), w.end()));
}

// -----------------------------------------------------------------------------
// Synthetic Control with covariate balancing
// Objective:
//   min_w  0.5*||X w - y||^2 + 0.5*kappa*||Z w - z||^2 + 0.5*lambda*||w||^2
//   s.t.   w >= 0, 1'w = 1
//
// Step size 1/L, with L ≈ s_max(X)^2 + kappa*s_max(Z)^2 + lambda
// -----------------------------------------------------------------------------

// [[Rcpp::export(name = ".sc_pg_simplex_cov")]]
Rcpp::NumericVector sc_pg_simplex_cov(const arma::mat &X,
                                      const arma::vec &y,
                                      const arma::mat &Z,
                                      const arma::vec &z,
                                      const double kappa = 1e-2,
                                      const double lambda = 1e-3,
                                      const int maxit = 5000,
                                      const double tol = 1e-8) {
  const arma::uword n = X.n_rows;
  if (Z.n_rows != n) Rcpp::stop("Z must have same number of rows (donors) as X");
  if (y.n_elem != X.n_cols) Rcpp::stop("y length mismatch with X columns");
  if (z.n_elem != Z.n_cols) Rcpp::stop("z length mismatch with Z columns");

  arma::vec w(n, arma::fill::ones);
  w /= static_cast<double>(n);

  // Lipschitz constant now accounts for the covariate term. Bounding each
  // spectral norm separately keeps the expression readable yet safe.
  arma::vec sx, sz;
  arma::mat U, V;
  arma::svd_econ(U, sx, V, X, 'l');
  arma::svd_econ(U, sz, V, Z, 'l');
  double L = (sx.n_elem > 0 ? sx.max() * sx.max() : 1.0)
    + kappa * (sz.n_elem > 0 ? sz.max() * sz.max() : 0.0)
    + lambda;
  double step = 1.0 / L;

  double f_old = arma::datum::inf;

  for (int it = 0; it < maxit; ++it) {
    arma::vec Xw = X * w;
    arma::vec Zw = Z * w;
    arma::vec grad = X.t() * (Xw - y)
      + kappa * Z.t() * (Zw - z)
      + lambda * w;

    arma::vec w_new = project_simplex(w - step * grad);

    double f_new = 0.5 * arma::dot(Xw - y, Xw - y)
      + 0.5 * kappa * arma::dot(Zw - z, Zw - z)
      + 0.5 * lambda * arma::dot(w, w);

    if (rel_improve(f_old, f_new) < tol) {
      w = w_new;
      break;
    }
    w = w_new;
    f_old = f_new;
  }

  return Rcpp::wrap(Rcpp::NumericVector(w.begin(), w.end()));
}
