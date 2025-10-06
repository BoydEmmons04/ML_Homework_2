# boston_linreg_hw2.py
# -----------------------------------------------------------------------------
# SINGLE-FILE PROJECT SCAFFOLD (COMMENTS-ONLY) — NO THIRD-PARTY ML LIBRARIES
# Copy/paste this into your file and fill in each TODO with your own code.
# -----------------------------------------------------------------------------
# GOALS
# - Load a tabular dataset (CSV).
# - Implement linear regression FROM SCRATCH (no numpy/scikit-learn).
# - Train via either:
#     (A) Closed-form Normal Equation (requires your own matrix ops), or
#     (B) Gradient Descent (batch) with your own math helpers.
# - Evaluate with MSE and R^2
# - (Optional) k-fold CV, standardization, polynomial features, residual plots.
# - Keep everything in this single file.
# -----------------------------------------------------------------------------


# =========================
# 0) RUNTIME/CLI OVERVIEW
# =========================
# TODO: Add argparse to accept:
#   --data PATH_TO_CSV
#   --target TARGET_COLUMN_NAME (default: MEDV or as needed)
#   --model ols|gd (closed-form vs gradient-descent)
#   --alpha FLOAT (learning rate for GD; default e.g. 0.01)
#   --epochs INT (for GD; default e.g. 5000)
#   --test-size FLOAT (e.g., 0.2)
#   --standardize (flag to z-score features using training stats)
#   --poly-degree INT (1 = no expansion; 2+ = add polynomial terms)
#   --cv INT (0 = no CV; k >= 2 = k-fold CV on train)
#   --seed INT (for reproducibility)
#   --no-plots (optional; skip residual/diagnostic plots)
#   --outdir PATH (where to save any figures/metrics)
#
# NOTE: Do not implement here yet; just reserve the structure above for later.


# =========================
# 1) DATA LOADING
# =========================
# TODO: Read CSV from --data using Python's csv module (no pandas).
#   - Parse header row to get column names.
#   - Convert each subsequent row to floats where possible.
#   - Handle missing values as required by your assignment (drop/zero/impute).
#   - Ensure the target column exists; separate features (X) and target (y).
#   - Return:
#       X_raw: list of lists (rows x features)
#       y_raw: list (rows,)
#       feature_names: list of str (excluding target)

def load_boston_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data_start = None
    for i, line in enumerate(lines):
        if line.strip() and (line.strip()[0].isdigit() or line.strip()[0] == '.'):
            data_start = i
            break

    if data_start is None:
        raise ValueError("No data found in the file.")

    data_lines = lines[data_start:]

    data = []
    row = []
    for line in data_lines:
        try:
            values = [float(x) for x in line.strip().split() if x]
        except ValueError:
            continue  # Skip lines that can't be converted
        if values:
            row.extend(values)
            # Boston dataset: 14 columns per row
            while len(row) >= 14:
                data.append(row[:14])
                row = row[14:]

    return data


# =========================
# 2) QUICK EDA (PRINTS ONLY)
# =========================
# TODO: Print basic info to stdout:
#   - Number of rows, number of columns
#   - Head (first N rows) as formatted text
#   - Per-column min/max/mean (compute with your own helpers)
#   - Count of missing/invalid entries (if any)
# NOTE: Keep it lightweight and text-only to avoid external libs.


# =========================
# 3) PREPROCESSING HELPERS (FROM SCRATCH)
# =========================
# TODO: Implement pure-Python helpers (no numpy):
#   - mean(values: list[float]) -> float
#   - stdev(values: list[float]) -> float  (population or sample; be consistent)
#   - zscore_column(col: list[float]) -> list[float]   (use train stats only)
#   - add_bias_column(X: list[list[float]]) -> list[list[float]]  (prepend 1.0)
#   - polynomial_expansion(row: list[float], degree: int) -> list[float]
#       (degree=1 returns original row; degree>=2 adds x_i^2, cross terms optional
#        depending on assignment requirements)
#   - train/test split (use --test-size and --seed)
#       split_xy(X, y, test_size, seed) -> X_train, X_test, y_train, y_test
#
# NOTE: For standardization:
#   - Compute means/stds on X_train columns only.
#   - Apply those stats to transform BOTH X_train and X_test.


# =========================
# 4) LINEAR ALGEBRA BUILDING BLOCKS (NO NUMPY)
# =========================
# TODO: Implement basic matrix ops using lists:
#   - mat_shape(A) -> (rows, cols)
#   - mat_transpose(A) -> A_T
#   - mat_mul(A, B) -> A @ B   (validate inner dims; write triple loop)
#   - vec_dot(u, v) -> float
#   - mat_vec_mul(A, v) -> w
#   - scalar_vec_mul(c, v) -> scaled vector
#   - vec_add(u, v) / vec_sub(u, v) -> elementwise
#   - OPTIONAL: mat_add, mat_sub if useful
#
# For OLS closed-form:
#   - You will need to solve (X^T X) w = X^T y
#   - Implement a linear system solver:
#       * Gaussian elimination with partial pivoting, or
#       * (Safer) Use gradient descent route to avoid matrix inversion.
#   - If you choose Normal Equation, implement:
#       normal_equation(X_with_bias, y) -> weights


# =========================
# 5) MODEL: ORDINARY LEAST SQUARES (TWO PATHS)
# =========================
# OLS via Closed-Form (Normal Equation)
# -------------------------------------
# TODO (Option A):
#   - Build X_b = add_bias_column(X_processed)
#   - Compute XT = transpose(X_b)
#   - Compute A = XT @ X_b
#   - Compute b = XT @ y
#   - Solve A w = b for w using your solver
#   - Return weights w (including bias as w[0])
#
# OLS via Gradient Descent
# ------------------------
# TODO (Option B):
#   - Initialize w (len = n_features + 1) to zeros or small random values
#   - For epoch in [1..epochs]:
#       * Predict y_pred = X_b @ w   (implement as mat_vec_mul)
#       * Residuals r = y_pred - y
#       * Compute gradient g:
#           g = (2/N) * (X_b^T @ r)     # derive using calculus
#       * Update w = w - alpha * g
#       * (Optional) Track train MSE each epoch; early stop if improvement < tol
#   - Return learned weights w


# =========================
# 6) PREDICTION & LOSS FUNCTIONS
# =========================
# TODO:
#   - predict(X_b, w) -> list[float]
#   - mse(y_true, y_pred) -> float
#   - r2(y_true, y_pred) -> float   (1 - SS_res/SS_tot; handle zero-variance y)


# =========================
# 7) TRAINING WORKFLOW
# =========================
# TODO:
#   - Accept CLI args
#   - Load data (X_raw, y_raw, feature_names)
#   - Split to train/test
#   - If --standardize: compute train means/stds, transform X_train, X_test
#   - If --poly-degree > 1: expand features consistently for train/test
#   - Build X_b for train/test (bias first)
#   - If --model == 'ols': use normal equation path
#   - If --model == 'gd' : use gradient descent path with (--alpha, --epochs)
#   - Store final weights


# =========================
# 8) EVALUATION & REPORTING
# =========================
# TODO:
#   - Compute y_pred_train, y_pred_test
#   - Compute and PRINT:
#       * Train MSE, Train R^2
#       * Test MSE, Test R^2
#   - Print learned weights with their associated feature names:
#       bias = w[0]
#       for i, name in enumerate(feature_names_expanded): print(name, w[i+1])
#   - (Optional) Save a small metrics.txt/json in --outdir


# =========================
# 9) K-FOLD CROSS-VALIDATION (OPTIONAL)
# =========================
# TODO (if --cv >= 2):
#   - Split training data into k folds using your own splitter (shuffle by seed).
#   - For each fold:
#       * Fit model on k-1 folds, evaluate on holdout
#       * Record fold MSE/R^2
#   - Print mean ± std across folds
#   - Use the same preprocessing policy:
#       * Fit standardization stats on the CURRENT TRAIN SPLIT ONLY
#       * Apply to that split’s train/val partitions consistently


# =========================
# 10) DIAGNOSTIC PLOTS (OPTIONAL, TEXT-ONLY ALTERNATIVE)
# =========================
# Without third-party plotting libs, you can:
#   - Print binned residual summaries (e.g., residual mean/std across quantiles)
#   - Or, if allowed to write rudimentary CSVs:
#       * Save pairs (y_true, y_pred, residual) to CSV for external plotting
# TODO:
#   - Create residuals_test = y_test - y_pred_test
#   - Print basic residual stats (mean ~ 0, spread, a few samples)
#   - (Optional) Write "diagnostics.csv" if --outdir provided


# =========================
# 11) ERROR HANDLING & EDGE CASES
# =========================
# TODO:
#   - Validate that target is numeric
#   - Handle constant columns (drop or keep—document choice)
#   - Handle singular A = X^T X (if using normal equation):
#       * Fallback to gradient descent OR
#       * Add tiny L2 (ridge-like) lambda on diagonal (document if allowed)
#   - Protect against zero division in standardization (std = 0)
#   - Ensure reproducible shuffles with --seed


# =========================
# 12) MAIN ENTRY POINT
# =========================
# TODO:
#   - Implement standard "if __name__ == '__main__':" guard
#   - Parse args, run the workflow, catch exceptions, exit with code != 0 on error
#   - Keep logs/prints concise and clearly labeled

def main():
    data = load_boston_txt('boston.txt')
    print (data[0:5])  # Print first 5 rows for verification

if __name__ == '__main__':
    main()


# =========================
# 13) MANUAL TEST PLAN (NO LIBS)
# =========================
# TODO:
#   - Create a tiny synthetic CSV (3–4 rows, 1–2 features) where you can solve
#     weights by hand; verify both OLS and GD match expected solution.
#   - Check standardization off/on yields consistent predictions (after inverse).
#   - Check polynomial_degree=1 vs 2 changes feature count as expected.
#   - Verify CV runs and reports sane averages.
#   - Confirm no external dependencies are required to run.


# =========================
# 14) SUBMISSION CHECKLIST
# =========================
# TODO:
#   - Single file, no external libs.
#   - Clear instructions at top for how to run (args).
#   - Printed metrics (train/test MSE & R^2).
#   - (Optional) diagnostics.csv and metrics.txt in outdir.
#   - Comments explain design choices (normal eq vs GD, standardization policy).
