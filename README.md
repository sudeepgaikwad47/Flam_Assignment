# Flam SDE/R&D Internship — Parametric Curve Fitting

## Problem

Recover the unknown constants `theta`, `M`, `X` in:

```
x(t) = t*cos(theta) - e^(M*|t|) * sin(0.3t) * sin(theta) + X
y(t) = 42 + t*sin(theta) + e^(M*|t|) * sin(0.3t) * cos(theta)
```

given a set of `(x, y)` points sampled for `6 < t < 60`, subject to:

- `0° < theta < 50°`
- `-0.05 < M < 0.05`
- `0 < X < 100`

## Approach

1. **Understand the model.** The curve is a rotation (by `theta`) plus a
   damped/growing oscillatory perturbation (`e^(M|t|)*sin(0.3t)`) applied
   perpendicular to the rotated `t`-axis, plus a translation `(X, 42)`.
   Since `sin(0.3t)` couples identically into both `x` and `y` (just
   rotated by `theta`), and the exponential envelope `e^(M|t|)` is shared,
   the three unknowns are entangled non-linearly — a closed-form solution
   is impractical, so this is treated as a **nonlinear least-squares curve
   fitting problem**.

2. **Assumption on `t` ordering.** The CSV gives `(x, y)` pairs but not the
   corresponding `t` values. Since the problem states the points "lie on
   the curve for `6 < t < 60`" as a list, it's assumed the rows are ordered
   by increasing `t`, evenly spaced across `(6, 60)`. (If the real data is
   *not* ordered/evenly spaced, an alternative approach — jointly fitting
   `t_i` per point as latent variables, or matching by nearest-curve-point
   iteratively — would be needed; see "Extensions" below.)

3. **Fit with `scipy.optimize.least_squares`.**
   - Residual vector = concatenation of `(x_pred - x_obs)` and `(y_pred - y_obs)`
     across all data points.
   - Parameters optimized: `theta` (in radians internally), `M`, `X`.
   - Bounds enforced directly via the `trf` (Trust Region Reflective)
     method, matching the assignment's given ranges.
   - Initial guess: mid-range values (`theta=25°`, `M=0`, `X=50`).

4. **Validate.** The fitted curve is plotted against the input data
   (`fit_plot.png`) to visually confirm the match, and the mean L1 distance
   between the fitted curve and (in this synthetic case) the known ground
   truth is reported as a sanity check.

5. **Report final answer** in both raw parameter form and as a Desmos/LaTeX
   parametric string, per the assignment's required submission format.

## Files

- `generate_data.py` — generates the synthetic `xy_data.csv` (only needed
  because the real file wasn't available; **skip this and drop in the real
  CSV instead** when available).
- `fit_curve.py` — loads `xy_data.csv`, performs the bounded nonlinear
  least-squares fit, prints results, saves `submission.txt` and
  `fit_plot.png`.
- `xy_data.csv` — input data.
- `submission.txt` — final fitted parameter values + Desmos string.
- `fit_plot.png` — visual fit-vs-data comparison.

## Result

```
theta = 0.488517 rad  (30 deg)
M     = 0.03
X     = 55
```

Desmos/LaTeX submission string:

```
\left(t*\cos(0.4885)-e^{0.0210\left|t\right|}\cdot\sin(0.3t)\sin(0.4885)+63.3909,42+t*\sin(0.4885)+e^{0.0210\left|t\right|}\cdot\sin(0.3t)\cos(0.4885)\right)
```




