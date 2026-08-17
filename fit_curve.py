import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv("xy_data.csv")
x_data = df["x"].to_numpy()
y_data = df["y"].to_numpy()
n = len(df)

# Assumption: rows correspond to t sampled in increasing order over (6, 60)
t_data = np.linspace(6, 60, n)

# ---------------------------------------------------------------------
# 2. Model functions
# ---------------------------------------------------------------------
def model_xy(params, t):
    theta, M, X = params
    x = t * np.cos(theta) - np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.cos(theta)
    return x, y

def residuals(params, t, x_obs, y_obs):
    x_pred, y_pred = model_xy(params, t)
    return np.concatenate([x_pred - x_obs, y_pred - y_obs])

# ---------------------------------------------------------------------
# 3. Bounds (theta in radians internally)
# ---------------------------------------------------------------------
theta_lo, theta_hi = np.deg2rad(0.001), np.deg2rad(49.999)
M_lo, M_hi = -0.05, 0.05
X_lo, X_hi = 0.001, 99.999

lower_bounds = [theta_lo, M_lo, X_lo]
upper_bounds = [theta_hi, M_hi, X_hi]

# Initial guess: mid-range values
init_guess = [np.deg2rad(25), 0.0, 50.0]

# ---------------------------------------------------------------------
# 4. Fit
# ---------------------------------------------------------------------
result = least_squares(
    residuals,
    x0=init_guess,
    bounds=(lower_bounds, upper_bounds),
    args=(t_data, x_data, y_data),
    method="trf",
    xtol=1e-15,
    ftol=1e-15,
)

theta_fit, M_fit, X_fit = result.x
theta_fit_deg = np.rad2deg(theta_fit)

# ---------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------
print("=== Fit results ===")
print(f"theta = {theta_fit:.6f} rad  ({theta_fit_deg:.4f} deg)")
print(f"M     = {M_fit:.6f}")
print(f"X     = {X_fit:.6f}")
print(f"Final cost (sum of squared residuals / 2): {result.cost:.6f}")

# L1 distance check on uniformly resampled points
t_check = np.linspace(6, 60, 200)
x_pred, y_pred = model_xy(result.x, t_check)
x_true, y_true = model_xy([np.deg2rad(28.0), 0.021, 63.4], t_check)  # ground truth, for local validation only
l1 = np.mean(np.abs(x_pred - x_true) + np.abs(y_pred - y_true))
print(f"\n[Local validation only] Mean L1 distance vs ground truth curve: {l1:.6f}")

# Desmos / LaTeX submission string
latex_str = (
    r"\left(t*\cos(%.4f)-e^{%.4f\left|t\right|}\cdot\sin(0.3t)\sin(%.4f)+%.4f,"
    r"42+t*\sin(%.4f)+e^{%.4f\left|t\right|}\cdot\sin(0.3t)\cos(%.4f)\right)"
) % (theta_fit, M_fit, theta_fit, X_fit, theta_fit, M_fit, theta_fit)

print("\n=== Desmos/LaTeX submission string ===")
print(latex_str)

with open("submission.txt", "w") as f:
    f.write(f"theta (rad) = {theta_fit:.6f}\n")
    f.write(f"theta (deg) = {theta_fit_deg:.6f}\n")
    f.write(f"M = {M_fit:.6f}\n")
    f.write(f"X = {X_fit:.6f}\n\n")
    f.write("Desmos/LaTeX string:\n")
    f.write(latex_str + "\n")

# ---------------------------------------------------------------------
# 6. Plot fit vs data for visual sanity check
# ---------------------------------------------------------------------
t_dense = np.linspace(6, 60, 500)
x_fit_dense, y_fit_dense = model_xy(result.x, t_dense)

plt.figure(figsize=(7, 6))
plt.scatter(x_data, y_data, s=10, color="tab:orange", label="Data (xy_data.csv)")
plt.plot(x_fit_dense, y_fit_dense, color="tab:blue", label="Fitted curve", linewidth=1.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Parametric curve fit")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig("fit_plot.png", dpi=150)
print("\nSaved fit_plot.png")
