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

namespace {

/*
 * weighted_row_means
 * ------------------
 * Computes row-wise weighted averages of a matrix while gracefully handling
 * missing entries (NaN). Each row i is treated independently: we iterate over
 * columns, skip any column whose entry or weight is not finite, accumulate the
 * numerator (value * weight), and track the sum of usable weights. If a row
 * ends up with zero total weight, we leave the output as NaN to signal that
 * the row contains no reliable information for the requested weights.
 */
arma::vec weighted_row_means(const arma::mat &M, const arma::vec &w) {
  arma::vec out(M.n_rows, arma::fill::value(arma::datum::nan));
  arma::uword cols = std::min<arma::uword>(M.n_cols, w.n_elem);
  if (cols == 0) return out;

  for (arma::uword i = 0; i < M.n_rows; ++i) {
    double total = 0.0, sumw = 0.0;
    for (arma::uword c = 0; c < cols; ++c) {
      double weight = w[c];
      double val = M(i, c);
      if (!std::isfinite(val) || !std::isfinite(weight) || weight <= 0.0) continue;
      total += weight * val;
      sumw += weight;
    }
    if (sumw > 0.0) out[i] = total / sumw;
  }
  return out;
}

/*
 * weighted_col_means
 * ------------------
 * Mirrors the previous helper but aggregates along columns, i.e. each column
 * receives a weighted average across rows. This is used when we collapse donor
 * matrices into a single synthetic path given unit-level weights. Missing
 * entries are skipped the same way as above.
 */
arma::vec weighted_col_means(const arma::mat &M, const arma::vec &v) {
  arma::vec out(M.n_cols, arma::fill::value(arma::datum::nan));
  arma::uword rows = std::min<arma::uword>(M.n_rows, v.n_elem);
  if (rows == 0) return out;

  for (arma::uword c = 0; c < M.n_cols; ++c) {
    double total = 0.0, sumw = 0.0;
    for (arma::uword r = 0; r < rows; ++r) {
      double weight = v[r];
      double val = M(r, c);
      if (!std::isfinite(val) || !std::isfinite(weight) || weight <= 0.0) continue;
      total += weight * val;
      sumw += weight;
    }
    if (sumw > 0.0) out[c] = total / sumw;
  }
  return out;
}

/*
 * finite_mean
 * -----------
 * Given a vector that may contain NaN placeholders, return the average of the
 * finite subset. If all entries are missing we propagate NaN to signal that
 * the caller does not have enough information.
 */
double finite_mean(const arma::vec &v) {
  arma::uvec idx = arma::find_finite(v);
  if (idx.is_empty()) return arma::datum::nan;
  arma::vec vals = v.elem(idx);
  return arma::mean(vals);
}

} // namespace

// Forward declaration: implemented in sc.cpp and linked at build time
Rcpp::NumericVector sc_pg_simplex(const arma::mat &X,
                                  const arma::vec &y,
                                  const double lambda = 1e-3,
                                  const int maxit = 5000,
                                  const double tol = 1e-8);

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
                          const double lambda_time,
                          const int maxit,
                          const double tol) {
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
      for (int tt : win)
        if (tt >= 1 && tt <= T_max && std::isfinite(Y[unit][tt])) ++obs;
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

    // Columns with any NaN break the synthetic control solvers (the SVD and
    // gradient calculations would produce NaNs). We therefore keep only the
    // subset of pre-treatment periods that are fully observed across both
    // treated and donor matrices.
    std::vector<arma::uword> pre_keep;
    pre_keep.reserve(Tpre);
    for (arma::uword c = 0; c < Tpre; ++c) {
      bool ok = true;
      for (arma::uword r = 0; r < Y_T_pre.n_rows; ++r) {
        if (!std::isfinite(Y_T_pre(r, c))) { ok = false; break; }
      }
      if (!ok) continue;
      for (arma::uword r = 0; r < Y_D_pre.n_rows; ++r) {
        if (!std::isfinite(Y_D_pre(r, c))) { ok = false; break; }
      }
      if (ok) pre_keep.push_back(c);
    }
    if (pre_keep.empty()) continue;

    arma::uvec pre_idx(pre_keep.size());
    for (std::size_t j = 0; j < pre_keep.size(); ++j) pre_idx[j] = pre_keep[j];

    // Unit weights (synthetic control) on the cleaned pre window. The target
    // path is the average treated trajectory, so the donor combination mimics
    // the treated group before treatment.
    arma::rowvec y_tgt = arma::mean(Y_T_pre.cols(pre_idx), 0);
    arma::vec v = Rcpp::as<arma::vec>(
      sc_pg_simplex(Y_D_pre.cols(pre_idx), y_tgt.t(), lambda_unit, maxit, tol)
    );

    // Time weights learned on the same subset of periods. We extend the vector
    // back to the full window length by filling zeros for dropped columns so
    // that downstream code can still align weights with matrices that carry
    // NAs in the excluded periods.
    arma::mat combined = arma::join_cols(Y_T_pre.cols(pre_idx), Y_D_pre.cols(pre_idx));
    arma::vec mu = arma::mean(combined, 0).t();
    arma::vec omega = arma::zeros<arma::vec>(Tpre);
    if (!pre_idx.is_empty()) {
      arma::vec omega_sub = Rcpp::as<arma::vec>(
        time_weights_pg(mu, lambda_time, maxit, tol)
      );
      omega.elem(pre_idx) = omega_sub;
    }

    // Collapse matrices into scalar summaries that respect both sets of weights
    // and continue to ignore missing cells.
    arma::vec treated_pre_means  = weighted_row_means(Y_T_pre,  omega);
    arma::vec treated_post_means = weighted_row_means(Y_T_post, omega);
    arma::vec donor_pre_means    = weighted_col_means(Y_D_pre,  v);
    arma::vec donor_post_means   = weighted_col_means(Y_D_post, v);

    double T_pre_hat_T  = finite_mean(treated_pre_means);
    double T_post_hat_T = finite_mean(treated_post_means);
    double C_pre_hat    = finite_mean(donor_pre_means);
    double C_post_hat   = finite_mean(donor_post_means);

    double tau = (T_post_hat_T - T_pre_hat_T) - (C_post_hat - C_pre_hat);
    tau_g_buf[gg_idx] = tau;
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
