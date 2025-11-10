// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
#include <RcppArmadillo.h>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include "utils.h"

/*
 * did.cpp
 * -------
 * This file implements a light-weight cohort-level Difference-in-Differences
 * estimator. The idea is to compute, for every treatment cohort g, the change
 * in average outcomes around treatment for treated units and compare it to
 * the change for donor units. The implementation mirrors the heavier SDID
 * routine but skips all synthetic weighting logic, which keeps the code short
 * and makes it a good place to understand the data manipulation steps.
 */

// ---------------------------------------------------------------------------
// did_cohort_att
//   id, t, y, g : stacked panel data in "long" format (1-based integer ids)
//   L, F        : number of pre- and post-treatment periods to include
//   min_cov     : minimum fraction of observed outcomes required in the
//                 pre-treatment window before we use a unit.
//
// The function returns a list containing the overall ATT (weighted by the
// number of treated units contributing per cohort) and a per-cohort table.
// ---------------------------------------------------------------------------
// [[Rcpp::export(name = ".did_cohort_att")]]
Rcpp::List did_cohort_att(const Rcpp::IntegerVector &id,
                          const Rcpp::IntegerVector &t,
                          const Rcpp::NumericVector &y,
                          const Rcpp::IntegerVector &g,
                          const int L, const int F, const double min_cov) {
  // Discover basic panel dimensions: number of stacked rows, and the largest
  // time/unit codes. The panel is assumed to be encoded with positive integers.
  int N = id.size();
  int T_max = 0; int I_max = 0;
  for (int k = 0; k < N; ++k) {
    if (t[k] > T_max) T_max = t[k];
    if (id[k] > I_max) I_max = id[k];
  }

  // Build a dense matrix-like container Y[i][tt] in 1-based indexing so we can
  // access outcomes quickly during the cohort loop. Missing cells stay NA.
  std::vector< std::vector<double> > Y(
    static_cast<std::size_t>(I_max) + 1,
    std::vector<double>(static_cast<std::size_t>(T_max) + 1, NA_REAL)
  );
  // Record each unit's cohort (first-treatment time). NA_INTEGER means never.
  std::vector<int> G_of(I_max+1, NA_INTEGER);

  for (int k = 0; k < N; ++k) {
    Y[id[k]][t[k]] = y[k];
    if (!Rcpp::IntegerVector::is_na(g[k])) G_of[id[k]] = g[k];
  }

  // Collect the unique cohort values in sorted order. We use an unordered_set
  // for the de-duplication, then copy into a vector we can index by position.
  std::unordered_set<int> uniq;
  for (int i = 1; i <= I_max; ++i)
    if (G_of[i] != NA_INTEGER) uniq.insert(G_of[i]);
  std::vector<int> Gs(uniq.begin(), uniq.end());
  std::sort(Gs.begin(), Gs.end());

  // Buffers to store cohort-specific effects and the number of treated units
  // used for each cohort. We keep them in plain std::vector so OpenMP can write
  // to different indices without touching R-managed memory.
  std::vector<double> tau_g_buf(Gs.size(), NA_REAL);
  std::vector<int> n_treated_buf(Gs.size(), 0);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int idx = 0; idx < (int)Gs.size(); ++idx) {
    int gg = Gs[idx];
    // Build the time windows. For example with L=2 we look at (g-2, g-1).
    std::vector<int> pre_win;
    for (int dt=L; dt>=1; --dt) pre_win.push_back(gg - dt);
    std::vector<int> post_win;
    for (int dt=0; dt<=F-1; ++dt) post_win.push_back(gg + dt);

    // Helper: shareable lambda that computes the fraction of observed outcomes
    // a unit has within a window. We guard against indexing outside [1, T_max].
    auto coverage = [&](int unit, const std::vector<int> &win){
      int obs = 0;
      for (int tt : win)
        if (tt>=1 && tt<=T_max && R_finite(Y[unit][tt])) ++obs;
      return static_cast<double>(obs) / win.size();
    };

    // Partition units into treated (cohort exactly gg) and donor (never-treated
    // or cohorts strictly after gg). We impose the coverage rule immediately.
    std::vector<int> treated_ids, donor_ids;
    for (int i = 1; i <= I_max; ++i) {
      int gi = G_of[i];
      bool is_treated = (gi == gg);
      bool is_donor = (Rcpp::IntegerVector::is_na(gi) || gi > gg);
      if (is_treated && coverage(i, pre_win) >= min_cov) treated_ids.push_back(i);
      if (is_donor && coverage(i, pre_win) >= min_cov) donor_ids.push_back(i);
    }
    if (treated_ids.empty() || donor_ids.empty()) continue;

    // Another helper: compute a simple average over a rectangular block
    // (subset of units × subset of periods) while skipping missing values.
    auto mean_over = [&](const std::vector<int> &units, const std::vector<int> &win){
      double s = 0.0; int c = 0;
      for (int i : units)
        for (int tt : win)
          if (tt>=1 && tt<=T_max && R_finite(Y[i][tt])) { s += Y[i][tt]; ++c; }
      return c>0 ? s/c : NA_REAL;
    };

    // Classical DiD estimator: difference in changes post vs pre.
    double Treated_pre = mean_over(treated_ids, pre_win);
    double Treated_post= mean_over(treated_ids, post_win);
    double Donor_pre = mean_over(donor_ids, pre_win);
    double Donor_post = mean_over(donor_ids, post_win);

    tau_g_buf[idx] = (Treated_post - Treated_pre) - (Donor_post - Donor_pre);
    n_treated_buf[idx] = static_cast<int>(treated_ids.size());
  }

  // Copy the thread-safe buffers back into Rcpp objects for the return value.
  Rcpp::NumericVector tau_g(Gs.size(), NA_REAL);
  Rcpp::IntegerVector n_treated(Gs.size(), 0);
  for (std::size_t i = 0; i < Gs.size(); ++i) {
    tau_g[i] = tau_g_buf[i];
    n_treated[i] = n_treated_buf[i];
  }

  // Weighted average of the cohort-specific effects. Weight = # treated units,
  // so larger cohorts contribute proportionally more to the final ATT.
  double num=0.0, den=0.0;
  for (int i=0;i<tau_g.size();++i)
    if (R_finite(tau_g[i])) { num+=tau_g[i]*n_treated[i]; den+=n_treated[i]; }
  double att = den>0 ? num/den : NA_REAL;

  return Rcpp::List::create(Rcpp::Named("att")=att,
                            Rcpp::Named("by_cohort")=Rcpp::DataFrame::create(Rcpp::Named("g")=Rcpp::wrap(Gs), Rcpp::Named("tau")=tau_g, Rcpp::Named("n_treated")=n_treated));
}
