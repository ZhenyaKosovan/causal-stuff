#pragma once

// ---------------------------------------------------------------------------
// Common utility helpers shared across the C++ translation units.
// Everything in this header is kept header-only so that each .cpp file can
// inline the tiny helpers without needing a separate compilation unit.
// ---------------------------------------------------------------------------

#include <RcppArmadillo.h>

// ---------------------------------------------------------------------------
// project_simplex
//   Given any real-valued vector `v`, returns the projection onto the
//   probability simplex { w : w_i >= 0, sum_i w_i = 1 }.
//
//   We implement the algorithm from Duchi et al. (2008):
//     1. Sort entries descending.
//     2. Find the largest prefix whose average shift keeps entries positive.
//     3. Subtract the threshold `theta` and clamp at zero.
//   The result is unique and preserves ordering, which is important for all of
//   the projected-gradient solvers used within this package.
// ---------------------------------------------------------------------------
inline arma::vec project_simplex(const arma::vec &v) {
  arma::vec u = arma::sort(v, "descend");     // descending copy for prefix sums
  arma::vec css = arma::cumsum(u);            // cumulative sum for theta search

  double theta = 0.0;                         // threshold that enforces sum=1
  for (arma::uword j = 0; j < u.n_elem; ++j) {
    double t = (css[j] - 1.0) / static_cast<double>(j + 1);
    // As soon as u_j - t <= 0 we have exceeded the active set.
    if (u[j] - t > 0.0) {
      theta = t;
    }
  }

  arma::vec w = v - theta;
  // Clamp all negative entries to zero; Armadillo transform applies a functor
  w.transform([](double x) { return x < 0.0 ? 0.0 : x; });

  double s = arma::accu(w);
  if (s == 0.0) {
    // Degenerate case: original vector was extremely negative. Return uniform.
    return arma::ones(v.n_elem) / static_cast<double>(v.n_elem);
  }
  return w / s;
}

// ---------------------------------------------------------------------------
// rel_improve
//   Measures the relative improvement between two objective function values.
//   We scale by max(1, |f_old|) so that early iterations (when f_old can be
//   zero) still give meaningful ratios. Positive return values mean progress;
//   small values indicate convergence.
// ---------------------------------------------------------------------------
inline double rel_improve(double f_old, double f_new) {
  double denom = std::max(1.0, std::abs(f_old));
  return (f_old - f_new) / denom;
}

#ifdef _OPENMP
#include <omp.h>
#endif
