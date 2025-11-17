// src/sdid.cpp
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
#include <RcppArmadillo.h>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <cmath>
#include "utils.h"

/*
 * sdid.cpp
 * --------
 * Houses the synthetic difference-in-differences (SDID) routines. Compared to
 * the simple DiD estimator, SDID builds donor weights (across units) and time
 * weights (across pre-treatment periods) so that the counterfactual trends of
 * treated units match the donor pool as closely as possible. Because the logic
 * touches many small helper calculations, we keep them in an anonymous
 * namespace for clarity and to give each piece detailed documentation.
 */

// Forward declaration: implemented in sc.cpp and linked at build time
Rcpp::NumericVector sc_pg_simplex(const arma::mat &X,
                                  const arma::vec &y,
                                  const double lambda = 1e-3,
                                  const int maxit = 5000,
                                  const double tol = 1e-8);

namespace {

double row_mean_ignore_nan(const arma::rowvec &row) {
  double total = 0.0;
  std::size_t count = 0;
  for (double val : row) {
    if (std::isfinite(val)) { total += val; ++count; }
  }
  if (count == 0) return arma::datum::nan;
  return total / static_cast<double>(count);
}

double matrix_mean_ignore_nan(const arma::mat &M) {
  double total = 0.0;
  std::size_t count = 0;
  for (double val : M) {
    if (std::isfinite(val)) { total += val; ++count; }
  }
  if (count == 0) return arma::datum::nan;
  return total / static_cast<double>(count);
}

double weighted_mean(const arma::vec &values, const arma::vec &weights) {
  double total = 0.0, sumw = 0.0;
  arma::uword n = std::min(values.n_elem, weights.n_elem);
  for (arma::uword i = 0; i < n; ++i) {
    double val = values[i];
    double w = weights[i];
    if (!std::isfinite(val) || !std::isfinite(w) || w <= 0.0) continue;
    total += w * val;
    sumw += w;
  }
  if (sumw <= 0.0) return arma::datum::nan;
  return total / sumw;
}

double weighted_dot(const arma::vec &w, const arma::rowvec &series) {
  double total = 0.0, sumw = 0.0;
  arma::uword n = std::min<arma::uword>(w.n_elem, series.n_elem);
  for (arma::uword i = 0; i < n; ++i) {
    double weight = w[i];
    double val = series[i];
    if (!std::isfinite(weight) || weight <= 0.0 || !std::isfinite(val)) continue;
    total += weight * val;
    sumw += weight;
  }
  if (sumw <= 0.0) return arma::datum::nan;
  return total / sumw;
}

arma::rowvec column_weighted_means(const arma::mat &M, const arma::vec &weights) {
  arma::rowvec out(M.n_cols, arma::fill::value(arma::datum::nan));
  for (arma::uword c = 0; c < M.n_cols; ++c) {
    double total = 0.0, sumw = 0.0;
    for (arma::uword r = 0; r < M.n_rows; ++r) {
      double val = M(r, c);
      double w = weights[r];
      if (!std::isfinite(val) || !std::isfinite(w) || w <= 0.0) continue;
      total += w * val;
      sumw += w;
    }
    if (sumw > 0.0) out[c] = total / sumw;
  }
  return out;
}

double donor_noise_sd(const arma::mat &donors) {
  if (donors.n_cols < 2 || donors.n_rows == 0) return 1.0;
  std::vector<double> diffs;
  diffs.reserve(donors.n_rows * (donors.n_cols - 1));
  for (arma::uword r = 0; r < donors.n_rows; ++r) {
    bool row_ok = true;
    for (arma::uword c = 1; c < donors.n_cols; ++c) {
      double prev = donors(r, c - 1);
      double cur = donors(r, c);
      if (!std::isfinite(prev) || !std::isfinite(cur)) { row_ok = false; break; }
      diffs.push_back(cur - prev);
    }
    if (!row_ok) continue;
  }
  if (diffs.size() < 2) return 1.0;
  double mean = 0.0;
  for (double v : diffs) mean += v;
  mean /= static_cast<double>(diffs.size());
  double var = 0.0;
  for (double v : diffs) {
    double delta = v - mean;
    var += delta * delta;
  }
  var /= static_cast<double>(diffs.size() - 1);
  if (!(var > 0.0) || !std::isfinite(var)) return 1.0;
  return std::sqrt(var);
}

arma::mat center_columns(const arma::mat &M) {
  arma::mat out = M;
  for (arma::uword c = 0; c < out.n_cols; ++c) {
    double total = 0.0;
    std::size_t count = 0;
    for (arma::uword r = 0; r < out.n_rows; ++r) {
      double val = out(r, c);
      if (std::isfinite(val)) { total += val; ++count; }
    }
    if (count == 0) continue;
    double mean = total / static_cast<double>(count);
    for (arma::uword r = 0; r < out.n_rows; ++r) {
      if (std::isfinite(out(r, c))) out(r, c) -= mean;
    }
  }
  return out;
}

arma::mat filter_rows(const arma::mat &Y) {
  std::vector<arma::uword> keep;
  for (arma::uword r = 0; r < Y.n_rows; ++r) {
    bool ok = true;
    for (arma::uword c = 0; c < Y.n_cols; ++c) {
      if (!std::isfinite(Y(r, c))) { ok = false; break; }
    }
    if (ok) keep.push_back(r);
  }
  arma::mat out(keep.size(), Y.n_cols);
  for (std::size_t i = 0; i < keep.size(); ++i) out.row(i) = Y.row(keep[i]);
  return out;
}

// Frank-Wolf Step for || Ax - b || ^2 + eta ||x||^2
// with X on unit simplex
arma::vec fw_step(const arma::mat &A,
                  const arma::vec &lambda,
                  const arma::vec &b,
                  const double eta) {
  arma::vec Ax = A * lambda;
  // compute 1/2 drad of the objective function
  arma::rowvec half_grad = (Ax - b).t() * A + eta * lambda.t();
  // find the index of the dimension where gradient is the smallest
  arma::uword idx = static_cast<arma::uword>(half_grad.index_min());
  // We found `idx` of the vertex we need to move to
  // Construct the direction
  arma::vec direction = -lambda;
  direction[idx] = 1.0 - lambda[idx];
  // If we have nowhere to move -> exit
  if (arma::all(direction == 0.0)) return lambda;

  // This does 1-dimensional minimization of f(lambda + gamma * d) and
  // projects it onto [0,1]
  arma::vec derr = A.col(idx) - Ax;
  // this is essentially construction of optimal unconstrained step
  // gamma* = <grad(f(lambda)), d> / [2(||Ad||^2 + eta ||d||^2)]
  double numerator = -arma::as_scalar(half_grad * direction);
  double denom = arma::dot(derr, derr) + eta * arma::dot(direction, direction);
  double step = denom > 0.0 ? numerator / denom : 0.0;
  // project step onto [0,1]
  double alpha = std::min(1.0, std::max(0.0, step));
  // update next point
  return lambda + alpha * direction;
}

arma::vec sparsify_weights(const arma::vec &w) {
  arma::vec out = w;
  double wmax = out.max();
  if (!(wmax > 0.0)) return out;
  double cutoff = wmax / 4.0;
  for (arma::uword i = 0; i < out.n_elem; ++i) {
    if (out[i] <= cutoff) out[i] = 0.0;
  }
  double sumw = arma::accu(out);
  if (sumw > 0.0) out /= sumw;
  return out;
}

arma::vec fw_optimize(const arma::mat &Y_aug,
                      const double zeta,
                      const double min_decrease,
                      const int maxit,
                      arma::vec lambda) {
  if (Y_aug.n_cols < 2) return arma::vec();
  arma::mat Y = center_columns(Y_aug);
  Y = filter_rows(Y);
  if (Y.n_rows == 0) return arma::vec();

  arma::mat A = Y.cols(0, Y.n_cols - 2);
  arma::vec b = Y.col(Y.n_cols - 1);
  arma::uword T0 = A.n_cols;

  if (lambda.n_elem != T0) {
    lambda.set_size(T0);
    lambda.fill(1.0 / static_cast<double>(T0));
  }

  double eta = static_cast<double>(A.n_rows) * zeta * zeta;
  double prev = arma::datum::inf;
  for (int it = 0; it < maxit; ++it) {
    lambda = fw_step(A, lambda, b, eta);
    arma::vec coeff(T0 + 1);
    coeff.head(T0) = lambda;
    coeff[T0] = -1.0;
    arma::vec err = Y * coeff;
    double val = zeta * zeta * arma::dot(lambda, lambda)
      + arma::dot(err, err) / static_cast<double>(A.n_rows);
    if (it >= 1 && (prev - val) <= (min_decrease * min_decrease)) break;
    prev = val;
  }
  return lambda;
}

arma::vec clean_weights(const arma::mat &Y_aug,
                        const double zeta,
                        const double min_decrease,
                        const int maxit,
                        const arma::vec &warm_start) {
  int pre_iters = std::min(100, maxit);
  if (pre_iters < 1) pre_iters = 1;
  arma::uword target = (Y_aug.n_cols > 0) ? (Y_aug.n_cols - 1) : 0;
  arma::vec init = warm_start;
  if (init.n_elem != target) init.reset();
  arma::vec warm = fw_optimize(Y_aug, zeta, min_decrease, pre_iters, init);
  arma::vec sparsified = sparsify_weights(warm);
  return fw_optimize(Y_aug, zeta, min_decrease, maxit, sparsified);
}

struct CohortFitResult {
  arma::vec unit_weights;
  arma::vec time_weights;
  double tau;
};

bool fit_sdid_cohort(const arma::mat &Y_T_pre,
                     const arma::mat &Y_D_pre,
                     const arma::mat &Y_T_post,
                     const arma::mat &Y_D_post,
                     const double lambda_unit,
                     const bool use_unit_default,
                     const double lambda_time,
                     const bool use_time_default,
                     const int maxit,
                     const double tol,
                     const double min_decrease_input,
                     const bool use_min_decrease_default,
                     const arma::vec &unit_warm,
                     const arma::vec &time_warm,
                     CohortFitResult &out) {
  if (Y_T_pre.n_cols == 0 || Y_D_pre.n_cols == 0 || Y_T_post.n_cols == 0 || Y_D_post.n_cols == 0)
    return false;

  arma::uword N0 = Y_D_pre.n_rows;
  arma::uword Tpre = Y_D_pre.n_cols;
  arma::uword N1 = Y_T_pre.n_rows;
  arma::uword Tpost = Y_D_post.n_cols;

  if (N0 == 0 || Tpre == 0 || Tpost == 0) return false;

  // Pre matrices must be fully observed; any NA/Inf should have been blocked in R.
  if (!Y_T_pre.is_finite() || !Y_D_pre.is_finite()) return false;

  double noise = donor_noise_sd(Y_D_pre);
  if (!std::isfinite(noise) || noise <= 0.0) noise = 1.0;
  double min_decrease = use_min_decrease_default ? (1e-5 * noise) : min_decrease_input;
  if (!std::isfinite(min_decrease) || min_decrease <= 0.0) min_decrease = 1e-5;

  double eta = std::pow(std::max(1.0, static_cast<double>(N1) * static_cast<double>(Tpost)), 0.25);
  if (!std::isfinite(eta) || eta <= 0.0) eta = 1.0;

  double zeta_unit = use_unit_default ? (eta * noise) : lambda_unit;
  double zeta_time = use_time_default ? (1e-6 * noise) : lambda_time;
  if (!std::isfinite(zeta_unit)) zeta_unit = eta * noise;
  if (!std::isfinite(zeta_time)) zeta_time = 1e-6 * noise;

  arma::mat collapsed(N0 + 1, Tpre + 1, arma::fill::value(arma::datum::nan));
  collapsed(arma::span(0, N0 - 1), arma::span(0, Tpre - 1)) = Y_D_pre;

  arma::vec donor_post_means(N0, arma::fill::value(arma::datum::nan));
  for (arma::uword r = 0; r < N0; ++r)
    donor_post_means[r] = row_mean_ignore_nan(Y_D_post.row(r));
  collapsed(arma::span(0, N0 - 1), Tpre) = donor_post_means;

  arma::rowvec treated_pre_means(Tpre, arma::fill::value(arma::datum::nan));
  for (arma::uword c = 0; c < Tpre; ++c)
    treated_pre_means[c] = row_mean_ignore_nan(Y_T_pre.col(c).t());
  collapsed.row(N0).cols(0, Tpre - 1) = treated_pre_means;

  double treated_post_mean = matrix_mean_ignore_nan(Y_T_post);
  collapsed(N0, Tpre) = treated_post_mean;
  if (!std::isfinite(treated_post_mean)) return false;

  arma::vec time_start;
  if (time_warm.n_elem == Y_D_pre.n_cols) time_start = time_warm;

  arma::vec time_sub = clean_weights(collapsed.rows(0, N0 - 1), zeta_time, min_decrease, maxit, time_start);
  if (time_sub.n_elem != Tpre) return false;

  arma::vec unit_start = unit_warm;
  if (unit_start.n_elem != N0) unit_start.reset();
  arma::vec unit_sub = clean_weights(collapsed.cols(0, Tpre - 1).t(), zeta_unit, min_decrease, maxit, unit_start);
  if (unit_sub.n_elem != N0) return false;

  arma::vec time_full = time_sub;

  arma::rowvec donor_pre = column_weighted_means(Y_D_pre, unit_sub);
  double donor_pre_hat = weighted_dot(time_sub, donor_pre);
  double treated_pre_hat = weighted_dot(time_sub, treated_pre_means);
  double donor_post_hat = weighted_mean(donor_post_means, unit_sub);

  if (!std::isfinite(donor_post_hat) || !std::isfinite(donor_pre_hat) || !std::isfinite(treated_pre_hat))
    return false;

  double tau = (treated_post_mean - treated_pre_hat) - (donor_post_hat - donor_pre_hat);

  out.unit_weights = unit_sub;
  out.time_weights = time_full;
  out.tau = tau;
  return std::isfinite(tau);
}

} // namespace

// -----------------------------------------------------------------------------
// Time weights for SDID via projected gradient on the simplex
// Smoothness objective around uniform weights:
//   min_omega  0.5*(omega - u)' H (omega - u)
// where H = B'B + lambda*I, B encodes first differences scaled by cross-sec means.
// -----------------------------------------------------------------------------

// [[Rcpp::export(name = ".time_weights_pg")]]
Rcpp::NumericVector time_weights_pg(const arma::vec &mu,
                                    const double lambda = 1e-3,
                                    const int maxit = 5000,
                                    const double tol = 1e-8) {
  const arma::uword T = mu.n_elem;
  if (T < 2) {
    // Degenerate case: only one time period. The simplex projection would do
    // the same, but returning early makes the intent explicit.
    arma::vec o = arma::ones(T) / static_cast<double>(T);
    return Rcpp::wrap(o);
  }

  arma::vec u = arma::ones(T) / static_cast<double>(T); // shrinkage target

  // Build (T-1) x T difference operator weighted by adjacent means.
  // Each row enforces smoothness between period i and i+1 by penalising their
  // difference scaled by the observed cross-sectional mean mu.
  arma::mat B(T - 1, T, arma::fill::zeros);
  for (arma::uword i = 0; i < T - 1; ++i) {
    B(i, i)     =  mu[i];
    B(i, i + 1) = -mu[i + 1];
  }
  arma::mat H = B.t() * B + lambda * arma::eye(T, T);

  // Fixed step size from Lipschitz constant (max eigenvalue of H). H is
  // positive semi-definite, so the largest singular value is enough.
  arma::vec s; arma::mat U, V;
  arma::svd_econ(U, s, V, H, 'l');
  double L = (s.n_elem > 0 ? s.max() : 1.0);
  double step = 1.0 / L;

  arma::vec w = u;
  double f_old = arma::datum::inf;

  for (int it = 0; it < maxit; ++it) {
    arma::vec grad = H * (w - u);                      // exact gradient
    arma::vec w_new = project_simplex(w - step * grad);
    double f_new = 0.5 * arma::as_scalar((w - u).t() * H * (w - u));

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
// Exported helpers
// -----------------------------------------------------------------------------

// [[Rcpp::export(name = ".sdid_fit_cohort")]]
Rcpp::List sdid_fit_cohort(const arma::mat &Y_T_pre,
                           const arma::mat &Y_D_pre,
                           const arma::mat &Y_T_post,
                           const arma::mat &Y_D_post,
                           const double lambda_unit,
                           const bool use_unit_default,
                           const double lambda_time,
                           const bool use_time_default,
                           const int maxit,
                           const double tol,
                           const double min_decrease,
                           const bool use_min_default,
                           const Rcpp::NumericVector &unit_warm,
                           const Rcpp::NumericVector &time_warm) {
  arma::vec unit_init(unit_warm.begin(), unit_warm.size());
  arma::vec time_init(time_warm.begin(), time_warm.size());
  CohortFitResult res;
  if (!fit_sdid_cohort(Y_T_pre, Y_D_pre, Y_T_post, Y_D_post,
                       lambda_unit, use_unit_default,
                       lambda_time, use_time_default,
                       maxit, tol, min_decrease, use_min_default,
                       unit_init, time_init, res)) {
    Rcpp::stop("Insufficient overlap to fit SDID weights");
  }
  return Rcpp::List::create(
    Rcpp::Named("v") = res.unit_weights,
    Rcpp::Named("omega") = res.time_weights,
    Rcpp::Named("tau") = res.tau
  );
}

// -----------------------------------------------------------------------------
// Panel-level SDID with cohort windows (OpenMP across cohorts)
// Inputs:
//   id, t : 1-based integer codes
//   y     : outcomes (can include NA)
//   g     : first-treatment time per unit (NA/large for never-treated)
//   L, F  : pre/post window sizes
// -----------------------------------------------------------------------------

// [[Rcpp::export(name = ".sdid_panel_att")]]
Rcpp::List sdid_panel_att(const Rcpp::IntegerVector &id,
                          const Rcpp::IntegerVector &t,
                          const Rcpp::NumericVector &y,
                          const Rcpp::IntegerVector &g,
                          const int L,
                          const int F,
                          const double min_cov,
                          const double lambda_unit,
                          const bool use_unit_default,
                          const double lambda_time,
                          const bool use_time_default,
                          const int maxit,
                          const double tol,
                          const double min_decrease,
                          const bool use_min_default) {
  const int N = id.size();
  if (!(N == t.size() && N == y.size() && N == g.size()))
    Rcpp::stop("Length mismatch among id, t, y, g");

  // Discover max unit/time codes. We use the same dense storage trick as in
  // did.cpp: allocate Y[i][t] with 1-based indices for O(1) lookups later.
  int T_max = 0, I_max = 0;
  for (int k = 0; k < N; ++k) {
    if (t[k] > T_max) T_max = t[k];
    if (id[k] > I_max) I_max = id[k];
  }

  std::vector< std::vector<double> >
    Y(static_cast<size_t>(I_max) + 1, std::vector<double>(static_cast<size_t>(T_max) + 1, NA_REAL));
  std::vector<int> G_of(static_cast<size_t>(I_max) + 1, NA_INTEGER);

  for (int k = 0; k < N; ++k) {
    Y[id[k]][t[k]] = y[k];
    if (!Rcpp::IntegerVector::is_na(g[k]) && G_of[id[k]] == NA_INTEGER)
      G_of[id[k]] = g[k];
  }

  // Unique cohort times
  std::unordered_set<int> uniq;
  for (int i = 1; i <= I_max; ++i) if (G_of[i] != NA_INTEGER) uniq.insert(G_of[i]);
  std::vector<int> Gs(uniq.begin(), uniq.end());
  std::sort(Gs.begin(), Gs.end());

  // As with did.cpp, the per-cohort contributions live in simple buffers so
  // that OpenMP can fill them safely.
  std::vector<double> tau_g_buf(Gs.size(), NA_REAL);
  std::vector<int> n_treated_buf(Gs.size(), 0);

  // Parallelize across cohorts
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int gg_idx = 0; gg_idx < static_cast<int>(Gs.size()); ++gg_idx) {
    int gg = Gs[gg_idx];

    // Build L-length pre windows and F-length post windows relative to gg.
    std::vector<int> pre_win; pre_win.reserve(L);
    std::vector<int> post_win; post_win.reserve(F);
    for (int dt = L; dt >= 1; --dt) pre_win.push_back(gg - dt);
    for (int dt = 0; dt <= F - 1; ++dt) post_win.push_back(gg + dt);

    // coverage helper
    auto coverage = [&](int unit, const std::vector<int> &win) {
      int obs = 0;
      for (int tt : win) {
        if (tt >= 1 && tt <= T_max && std::isfinite(Y[unit][tt])) ++obs;
      }
      return static_cast<double>(obs) / static_cast<double>(win.size());
    };

    // treated and donor sets
    std::vector<int> treated_ids; treated_ids.reserve(I_max);
    std::vector<int> donor_ids;   donor_ids.reserve(I_max);

    for (int i = 1; i <= I_max; ++i) {
      int gi = G_of[i];
      bool is_treated = (gi == gg);
      bool is_donor   = (Rcpp::IntegerVector::is_na(gi) || gi > gg);
      if (is_treated && coverage(i, pre_win) >= min_cov) treated_ids.push_back(i);
      if (is_donor   && coverage(i, pre_win) >= min_cov) donor_ids.push_back(i);
    }
    if (treated_ids.empty() || donor_ids.empty()) continue;

    const arma::uword Tpre  = static_cast<arma::uword>(pre_win.size());
    const arma::uword Tpost = static_cast<arma::uword>(post_win.size());

    // Assemble stacked matrices: rows = units, columns = periods in window.
    // We pre-fill everything with NaN so later logic can detect coverage gaps.
    arma::mat Y_T_pre(treated_ids.size(), Tpre);
    arma::mat Y_D_pre(donor_ids.size(),   Tpre);
    arma::mat Y_T_post(treated_ids.size(), Tpost);
    arma::mat Y_D_post(donor_ids.size(),   Tpost);
    Y_T_pre.fill(arma::datum::nan);
    Y_D_pre.fill(arma::datum::nan);
    Y_T_post.fill(arma::datum::nan);
    Y_D_post.fill(arma::datum::nan);

    for (arma::uword r = 0; r < treated_ids.size(); ++r) {
      int i = treated_ids[r];
      for (arma::uword c = 0; c < Tpre;  ++c) { int tt = pre_win[c];  if (tt>=1 && tt<=T_max)  Y_T_pre(r,c)  = Y[i][tt]; }
      for (arma::uword c = 0; c < Tpost; ++c) { int tt = post_win[c]; if (tt>=1 && tt<=T_max)  Y_T_post(r,c) = Y[i][tt]; }
    }
    for (arma::uword r = 0; r < donor_ids.size(); ++r) {
      int i = donor_ids[r];
      for (arma::uword c = 0; c < Tpre;  ++c) { int tt = pre_win[c];  if (tt>=1 && tt<=T_max)  Y_D_pre(r,c)  = Y[i][tt]; }
      for (arma::uword c = 0; c < Tpost; ++c) { int tt = post_win[c]; if (tt>=1 && tt<=T_max)  Y_D_post(r,c) = Y[i][tt]; }
    }

    CohortFitResult res;
    if (!fit_sdid_cohort(Y_T_pre, Y_D_pre, Y_T_post, Y_D_post,
                         lambda_unit, use_unit_default,
                         lambda_time, use_time_default,
                         maxit, tol, min_decrease, use_min_default,
                         arma::vec(), arma::vec(), res)) {
      continue;
    }
    tau_g_buf[gg_idx] = res.tau;
    n_treated_buf[gg_idx] = static_cast<int>(treated_ids.size());
  }

  // Aggregate ATT across cohorts, weighted by number of treated units used
  Rcpp::NumericVector tau_g(Gs.size(), NA_REAL);
  Rcpp::IntegerVector n_treated(Gs.size(), 0);
  for (std::size_t i = 0; i < Gs.size(); ++i) {
    tau_g[i] = tau_g_buf[i];
    n_treated[i] = n_treated_buf[i];
  }

  double num = 0.0, den = 0.0;
  for (int i = 0; i < tau_g.size(); ++i) {
    if (std::isfinite(tau_g[i])) { num += tau_g[i] * n_treated[i]; den += n_treated[i]; }
  }
  double att = (den > 0.0) ? (num / den) : NA_REAL;

  return Rcpp::List::create(
    Rcpp::Named("att") = att,
    Rcpp::Named("by_cohort") =
      Rcpp::DataFrame::create(Rcpp::Named("g") = Rcpp::wrap(Gs),
                              Rcpp::Named("tau") = tau_g,
                              Rcpp::Named("n_treated") = n_treated)
  );
}
