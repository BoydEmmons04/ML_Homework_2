
# boston_linreg_hw2.py
# -----------------------------------------------------------------------------
# Linear Regression HW 2 (minimal solution)
# Names : Boyd Emmons, Carter Ward
# Date  : Oct 7 2025
# Purpose: Predict MEDV using multivariable linear regression on Boston dataset.
# Rules matched:
#  - Train = first N-50 rows; Validation = last 50 rows
#  - Part-1 (GD): (2a) AGE+TAX; (2b) all predictors (except MEDV)
#  - For GD, standardize X using TRAIN means/stds; apply to validation
#  - Part-2 (Normal Eq): ONLY (2a), on UN-SCALED features with intercept
#  - Produce output.txt containing thetas and mean squared errors
# -----------------------------------------------------------------------------

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
    data, row = [], []
    for line in data_lines:
        try:
            values = [float(x) for x in line.strip().split() if x]
        except ValueError:
            continue
        if values:
            row.extend(values)
            while len(row) >= 14:
                data.append(row[:14])
                row = row[14:]
    return data

# ---------------- Basic stats / standardization (no numpy) -------------------

def mean(vals):
    s, n = 0.0, 0
    for v in vals:
        s += v; n += 1
    return s / n if n else 0.0

def stdev(vals):
    m = mean(vals)
    s, n = 0.0, 0
    for v in vals:
        d = v - m
        s += d * d
        n += 1
    return (s / n) ** 0.5 if n else 0.0

def fit_standardizer(X_cols):
    mus  = [mean(col) for col in X_cols]
    sigs = [stdev(col) for col in X_cols]
    sigs = [s if s != 0.0 else 1.0 for s in sigs]
    return mus, sigs

def apply_standardizer(X_rows, mus, sigs):
    Z = []
    for r in X_rows:
        Z.append([(r[j] - mus[j]) / sigs[j] for j in range(len(r))])
    return Z

def add_bias_column(X_rows):
    return [[1.0] + row for row in X_rows]

# ------------------- Linear algebra (lists) & solvers ------------------------

def mat_shape(A):
    return (len(A), len(A[0]) if A else 0)

def mat_transpose(A):
    r, c = mat_shape(A)
    return [[A[i][j] for i in range(r)] for j in range(c)]

def mat_vec_mul(A, v):
    r, c = mat_shape(A)
    out = []
    for i in range(r):
        s = 0.0
        for j in range(c):
            s += A[i][j] * v[j]
        out.append(s)
    return out

def mat_mul(A, B):
    rA, cA = mat_shape(A)
    rB, cB = mat_shape(B)
    assert cA == rB
    C = [[0.0]*cB for _ in range(rA)]
    for i in range(rA):
        for k in range(cA):
            aik = A[i][k]
            if aik == 0.0:
                continue
            for j in range(cB):
                C[i][j] += aik * B[k][j]
    return C

def solve_linear_system(A_in, b_in, ridge_lambda=0.0):
    n = len(A_in)
    A = [row[:] for row in A_in]
    b = b_in[:]
    if ridge_lambda != 0.0:
        for i in range(n):
            A[i][i] += ridge_lambda
    # Forward elimination with partial pivoting
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(A[r][col]))
        if abs(A[pivot][col]) < 1e-12:
            if ridge_lambda == 0.0:
                return solve_linear_system(A_in, b_in, ridge_lambda=1e-8)
            raise ValueError("Singular matrix.")
        if pivot != col:
            A[col], A[pivot] = A[pivot], A[col]
            b[col], b[pivot] = b[pivot], b[col]
        piv = A[col][col]
        for r in range(col+1, n):
            if A[r][col] == 0.0: continue
            f = A[r][col] / piv
            for c in range(col, n):
                A[r][c] -= f * A[col][c]
            b[r] -= f * b[col]
    # Back substitution
    x = [0.0]*n
    for i in reversed(range(n)):
        s = b[i]
        for j in range(i+1, n):
            s -= A[i][j]*x[j]
        x[i] = s / A[i][i]
    return x

# ------------------------ Loss / metrics -------------------------------------

def mse(y_true, y_pred):
    n = len(y_true)
    if n == 0: return 0.0
    s = 0.0
    for i in range(n):
        d = y_pred[i] - y_true[i]
        s += d*d
    return s / n

# -------------------- Models: GD (multi) & Normal Eq (2a) --------------------

def gradient_descent(Xb, y, alpha=0.01, epochs=5000):
    n, d = mat_shape(Xb)
    w = [0.0]*d
    XT = mat_transpose(Xb)
    two_over_n = 2.0 / n
    for _ in range(epochs):
        y_pred = mat_vec_mul(Xb, w)
        r = [y_pred[i] - y[i] for i in range(n)]
        g = mat_vec_mul(XT, r)
        for j in range(d):
            w[j] -= alpha * two_over_n * g[j]
    return w

def normal_equation_weights(Xb, y):
    XT = mat_transpose(Xb)
    A  = mat_mul(XT, Xb)        # (d x d)
    b  = mat_vec_mul(XT, y)     # (d)
    w  = solve_linear_system(A, b, ridge_lambda=1e-10)
    return w

# ------------------------------- Main ----------------------------------------

def main():
    # Load full dataset
    data = load_boston_txt('boston.txt')
    # Column names in order (Boston)
    names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
    AGE, TAX, MEDV = 6, 9, 13

    N = len(data)
    assert N >= 50, "Dataset too small."
    train = data[:N-50]
    val   = data[N-50:]

    # ---------------- Part-1: Gradient Descent ----------------
    # 2a: MEDV ~ AGE + TAX  (standardize AGE,TAX on TRAIN only)
    Xtr_2a = [[r[AGE], r[TAX]] for r in train]
    ytr    = [r[MEDV] for r in train]
    Xva_2a = [[r[AGE], r[TAX]] for r in val]
    yva    = [r[MEDV] for r in val]

    cols = [list(col) for col in zip(*Xtr_2a)]
    mus, sigs = fit_standardizer(cols)
    Xtr_2a_z = apply_standardizer(Xtr_2a, mus, sigs)
    Xva_2a_z = apply_standardizer(Xva_2a, mus, sigs)
    Xb_tr_2a = add_bias_column(Xtr_2a_z)
    Xb_va_2a = add_bias_column(Xva_2a_z)

    w_gd_2a  = gradient_descent(Xb_tr_2a, ytr, alpha=0.01, epochs=5000)
    yhat_va_2a_gd = mat_vec_mul(Xb_va_2a, w_gd_2a)
    mse_2a_gd = mse(yva, yhat_va_2a_gd)

    # 2b: MEDV ~ all predictors (13 cols except MEDV), standardize on TRAIN
    feat_idx = [i for i in range(14) if i != MEDV]
    Xtr_2b = [[r[i] for i in feat_idx] for r in train]
    Xva_2b = [[r[i] for i in feat_idx] for r in val]

    cols_all = [list(col) for col in zip(*Xtr_2b)]
    mus_all, sigs_all = fit_standardizer(cols_all)
    Xtr_2b_z = apply_standardizer(Xtr_2b, mus_all, sigs_all)
    Xva_2b_z = apply_standardizer(Xva_2b, mus_all, sigs_all)
    Xb_tr_2b = add_bias_column(Xtr_2b_z)
    Xb_va_2b = add_bias_column(Xva_2b_z)

    w_gd_2b  = gradient_descent(Xb_tr_2b, ytr, alpha=0.01, epochs=5000)
    yhat_va_2b_gd = mat_vec_mul(Xb_va_2b, w_gd_2b)
    mse_2b_gd = mse(yva, yhat_va_2b_gd)

    # ---------------- Part-2: Normal Equation (ONLY 2a) ----------------
    # Per spec: you don't need to scale features; include intercept.
    Xtr_2a_raw = add_bias_column(Xtr_2a)  # bias + [AGE, TAX]
    Xva_2a_raw = add_bias_column(Xva_2a)

    w_ne_2a = normal_equation_weights(Xtr_2a_raw, ytr)
    yhat_va_2a_ne = mat_vec_mul(Xva_2a_raw, w_ne_2a)
    mse_2a_ne = mse(yva, yhat_va_2a_ne)

    # ---------------- Output file (required) ----------------
    with open("output.txt", "w") as f:
        f.write("Part-1 (Gradient Descent)\n")
        f.write("Case 2a: MEDV ~ AGE_z + TAX_z (bias first)\n")
        f.write("theta_gd_2a = [" + ", ".join(f"{v:.8f}" for v in w_gd_2a) + "]\n")
        f.write(f"MSE_val_2a_gd = {mse_2a_gd:.8f}\n\n")

        f.write("Case 2b: MEDV ~ ALL_FEATURES_z (bias first)\n")
        f.write("theta_gd_2b = [" + ", ".join(f"{v:.8f}" for v in w_gd_2b) + "]\n")
        f.write(f"MSE_val_2b_gd = {mse_2b_gd:.8f}\n\n")

        f.write("Part-2 (Normal Equation)\n")
        f.write("Case 2a: MEDV ~ [1, AGE, TAX] (UNSCALED)\n")
        f.write("theta_ne_2a = [" + ", ".join(f"{v:.8f}" for v in w_ne_2a) + "]\n")
        f.write(f"MSE_val_2a_ne = {mse_2a_ne:.8f}\n")

    # Minimal console summary
    print("Saved results to output.txt")
    print(f"2a GD  MSE (val): {mse_2a_gd:.6f}")
    print(f"2b GD  MSE (val): {mse_2b_gd:.6f}")
    print(f"2a NE  MSE (val): {mse_2a_ne:.6f}")

if __name__ == '__main__':
    main()

