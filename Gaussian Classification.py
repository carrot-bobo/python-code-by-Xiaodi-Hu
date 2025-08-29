#Figure 28
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

# Generate synthetic data from a complex latent function
np.random.seed(42)
X = np.linspace(-6, 6, 120).reshape(-1, 1)

# True latent function: linear trend + oscillations
f_true = 0.5 * X.ravel() + np.sin(X).ravel() + 0.3 * np.cos(3 * X).ravel()

# Map latent values to probabilities via logistic link
p_true = 1.0 / (1.0 + np.exp(-f_true))

# Sample binary labels according to the probabilities
y = np.random.binomial(1, p_true)

# Split into train and test sets
n_train = 80
X_train, y_train = X[:n_train], y[:n_train]
X_test,  y_test  = X[n_train:], y[n_train:]

# Define and fit GPC with an RBF kernel
kernel = ConstantKernel(1.0) * RBF(length_scale=1.5)
gpc = GaussianProcessClassifier(kernel=kernel, random_state=42)
gpc.fit(X_train, y_train)

# Predict on the test set
proba_test = gpc.predict_proba(X_test)[:, 1]
y_pred = (proba_test >= 0.5).astype(int)

# Evaluate performance
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, proba_test)
brier = brier_score_loss(y_test, proba_test)
print(f"Accuracy = {acc:.4f}, AUC = {auc:.4f}, Brier = {brier:.4f}")

# Plot
plt.figure(figsize=(8, 5))

plt.scatter(
    X_train, y_train,
    color="blue",
    edgecolors="k",
    label="Training data",
    alpha=0.7
)

plt.scatter(
    X_test, y_test,
    color="blue",
    marker="x",
    label="Test data",
    alpha=0.9
)

plt.plot(
    X, p_true,
    "r--", lw=2,
    label="True probability"
)

proba_all = gpc.predict_proba(X)[:, 1]
plt.plot(
    X, proba_all,
    "b-", lw=2,
    label="GPC predicted probability"
)

plt.xlabel("x")
plt.ylabel("P(y=1 | x)")
plt.title("GPC on a Complex Synthetic Function")
plt.ylim(-0.05, 1.05)
plt.legend()
plt.tight_layout()
plt.show()


#-------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap 

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
    confusion_matrix, classification_report
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    RBF, RationalQuadratic, Matern, ExpSineSquared,
    DotProduct, WhiteKernel, ConstantKernel
)


DATA_PATH = "E:/系统默认/桌面/eggprice.csv"
N_LAGS    = 6
TEST_SIZE = 105
TAU_MODE  = "median"   
H_LIST    = [1, 4, 8]

#Load data
df = pd.read_csv(DATA_PATH)
if "price" in df.columns:
    prices = df["price"].to_numpy(dtype=float)
else:
    if df.shape[1] < 2:
        raise ValueError("CSV 需包含 'price' 列，或至少两列（价格在第2列）。")
    prices = df.iloc[:, 1].to_numpy(dtype=float)

T = len(prices)

# Fit and evaluate
def run_gpc_for_h(prices, h, n_lags=N_LAGS, test_size=TEST_SIZE, tau_mode=TAU_MODE, seed=42):
    if T <= (n_lags + h + test_size - 1):
        raise ValueError(f"h={h} 样本不足。")

    # Construct features and future prices
    X_list, p_future_list, t_index = [], [], []
    for t in range(n_lags, T - h):
        X_list.append(prices[t - n_lags:t][::-1])
        p_future_list.append(prices[t + h])
        t_index.append(t)

    X_all = np.vstack(X_list)
    p_future_all = np.asarray(p_future_list)
    t_index = np.asarray(t_index)

    # Time-based split
    n_all   = X_all.shape[0]
    n_train = n_all - test_size
    X_train_raw, X_test_raw = X_all[:n_train], X_all[n_train:]
    p_train_fut, p_test_fut = p_future_all[:n_train], p_future_all[n_train:]
    t_train, t_test = t_index[:n_train], t_index[n_train:]

    # Threshold and labels
    if isinstance(tau_mode, (int, float)):
        tau = float(tau_mode)
    elif tau_mode == "median":
        tau = float(np.median(p_train_fut))
    elif tau_mode == "mean":
        tau = float(np.mean(p_train_fut))
    else:
        raise ValueError("TAU_MODE 只能为 'median' | 'mean' | 数值")

    y_train = (p_train_fut > tau).astype(int)
    y_test  = (p_test_fut  > tau).astype(int)

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)


#-----------------------------------------------------------------------------
#Figure 30
    # Kernel：RBF + WhiteKernel(jitter) 
    def make_kernel(jitter):
        return (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
            + WhiteKernel(noise_level=jitter, noise_level_bounds=(1e-10, 1e-2))
        )

    jitters = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    last_err = None
    for jit in jitters:
        kernel = make_kernel(jit)
        gpc = GaussianProcessClassifier(
            kernel=kernel,
            random_state=seed,
            max_iter_predict=300,
            optimizer="fmin_l_bfgs_b"
        )
        try:
            gpc.fit(X_train, y_train)
            break
        except Exception as e:
            last_err = e
            continue
    else:
        raise RuntimeError(
            f"GPC fit failed under RBF even with jitter; last error:\n{last_err}"
        )

    # Prediction and evaluation
    proba_test = gpc.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= 0.5).astype(int)

    acc   = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, proba_test)
    auc   = roc_auc_score(y_test, proba_test) if np.any(y_test==0) and np.any(y_test==1) else np.nan

    print("="*70)
    print(f"h = {h} | 训练集: {n_train} | 测试集: {test_size} | τ = {tau:.6g}")
    print(f"Optimized kernel (full): {gpc.kernel_}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Brier    : {brier:.4f}")
    print(f"ROC-AUC  : {auc:.4f}" if not np.isnan(auc) else "ROC-AUC : N/A")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

    kernel_str = str(gpc.kernel_).split("+ WhiteKernel")[0].strip()
    kernel_str_wrapped = "Optimized kernel:\n" + "\n".join(textwrap.wrap(kernel_str, width=40))

    return {
        "t_test": t_test,
        "proba_test": proba_test,
        "y_test": y_test,
        "kernel_str": kernel_str_wrapped
    }

#  h=1,4,8
results = [run_gpc_for_h(prices, h) for h in H_LIST]

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, h, res in zip(axes, H_LIST, results):
    ax.plot(res["t_test"], res["proba_test"], lw=1.5, label="Predicted P(y=1)")
    ax.scatter(res["t_test"], res["y_test"], s=18, alpha=0.6, label="True label (0/1)")
    ax.axhline(0.5, ls="--", label="0.5 threshold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time index")
    ax.set_title(f"h = {h}\n{res['kernel_str']}", fontsize=9)

axes[0].set_ylabel("Probability / Label")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("GPC (RBF): Test-set Probabilities with Optimized Kernel (noise hidden)", y=1.05)
fig.tight_layout()
plt.show()


#------------------------------------------------------------------------------------------

#Figure 31
    # RQ + WhiteKernel(jitter) 
    def make_kernel(jitter):
        return (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RationalQuadratic(
                length_scale=1.0, alpha=1.0,
                length_scale_bounds=(1e-2, 1e3),
                alpha_bounds=(1e-2, 1e3)
            )
            + WhiteKernel(noise_level=jitter, noise_level_bounds=(1e-10, 1e-2))
        )

    jitters = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    last_err = None
    for jit in jitters:
        kernel = make_kernel(jit)
        gpc = GaussianProcessClassifier(
            kernel=kernel,
            random_state=seed,
            max_iter_predict=300,
            optimizer="fmin_l_bfgs_b"
        )
        try:
            gpc.fit(X_train, y_train)
            break
        except Exception as e:
            last_err = e
            continue
    else:
        raise RuntimeError(
            f"GPC fit failed under RQ even with jitter; last error:\n{last_err}"
        )

    proba_test = gpc.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= 0.5).astype(int)

    acc   = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, proba_test)
    auc   = roc_auc_score(y_test, proba_test) if np.any(y_test==0) and np.any(y_test==1) else np.nan

    print("="*70)
    print(f"h = {h} | 训练集: {n_train} | 测试集: {test_size} | τ = {tau:.6g}")
    print(f"Optimized kernel (full): {gpc.kernel_}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Brier    : {brier:.4f}")
    print(f"ROC-AUC  : {auc:.4f}" if not np.isnan(auc) else "ROC-AUC : N/A")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

    kernel_str = str(gpc.kernel_).split("+ WhiteKernel")[0].strip()
    kernel_str_wrapped = "Optimized kernel:\n" + "\n".join(textwrap.wrap(kernel_str, width=70))

    return {
        "t_test": t_test,
        "proba_test": proba_test,
        "y_test": y_test,
        "kernel_str": kernel_str_wrapped
    }

results = [run_gpc_for_h(prices, h) for h in H_LIST]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, h, res in zip(axes, H_LIST, results):
    ax.plot(res["t_test"], res["proba_test"], lw=1.5, label="Predicted P(y=1)")
    ax.scatter(res["t_test"], res["y_test"], s=18, alpha=0.6, label="True label (0/1)")
    ax.axhline(0.5, ls="--", label="0.5 threshold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time index")
    ax.set_title(f"h = {h}\n{res['kernel_str']}", fontsize=9)

axes[0].set_ylabel("Probability / Label")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("GPC (RQ): Test-set Probabilities with Optimized Kernel (noise hidden)", y=1.05)
fig.tight_layout()
plt.show()

#-----------------------------------------------------------------------------------------

#Figure 32
    #Matern + WhiteKernel(jitter) 
    def make_kernel(jitter):
        return (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(
                length_scale=1.0,
                length_scale_bounds=(1e-2, 1e3),
                nu=1.5  # 可改成 0.5, 2.5
            )
            + WhiteKernel(noise_level=jitter, noise_level_bounds=(1e-10, 1e-2))
        )

    jitters = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    last_err = None
    for jit in jitters:
        kernel = make_kernel(jit)
        gpc = GaussianProcessClassifier(
            kernel=kernel,
            random_state=seed,
            max_iter_predict=300,
            optimizer="fmin_l_bfgs_b"
        )
        try:
            gpc.fit(X_train, y_train)
            break
        except Exception as e:
            last_err = e
            continue
    else:
        raise RuntimeError(
            f"GPC fit failed under Matern even with jitter; last error:\n{last_err}"
        )

    proba_test = gpc.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= 0.5).astype(int)

    acc   = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, proba_test)
    auc   = roc_auc_score(y_test, proba_test) if np.any(y_test==0) and np.any(y_test==1) else np.nan

    print("="*70)
    print(f"h = {h} | 训练集: {n_train} | 测试集: {test_size} | τ = {tau:.6g}")
    print(f"Optimized kernel (full): {gpc.kernel_}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Brier    : {brier:.4f}")
    print(f"ROC-AUC  : {auc:.4f}" if not np.isnan(auc) else "ROC-AUC : N/A")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

    kernel_str = str(gpc.kernel_).split("+ WhiteKernel")[0].strip()
    kernel_str_wrapped = "Optimized kernel:\n" + "\n".join(textwrap.wrap(kernel_str, width=60))

    return {
        "t_test": t_test,
        "proba_test": proba_test,
        "y_test": y_test,
        "kernel_str": kernel_str_wrapped
    }
results = [run_gpc_for_h(prices, h) for h in H_LIST]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, h, res in zip(axes, H_LIST, results):
    ax.plot(res["t_test"], res["proba_test"], lw=1.5, label="Predicted P(y=1)")
    ax.scatter(res["t_test"], res["y_test"], s=18, alpha=0.6, label="True label (0/1)")
    ax.axhline(0.5, ls="--", label="0.5 threshold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time index")
    ax.set_title(f"h = {h}\n{res['kernel_str']}", fontsize=9)

axes[0].set_ylabel("Probability / Label")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("GPC (Matern): Test-set Probabilities with Optimized Kernel (noise hidden)", y=1.05)
fig.tight_layout()
plt.show()


#---------------------------------------------------------------------------------------------------

#Figure33
    # LIN + WhiteKernel(jitter) 
    def make_kernel(jitter):
        return (
            ConstantKernel(1.0, (1e-3, 1e3))
            * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
            + WhiteKernel(noise_level=jitter, noise_level_bounds=(1e-10, 1e-2))
        )

    jitters = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    last_err = None
    for jit in jitters:
        kernel = make_kernel(jit)
        gpc = GaussianProcessClassifier(
            kernel=kernel,
            random_state=seed,
            max_iter_predict=300,
            optimizer="fmin_l_bfgs_b"
        )
        try:
            gpc.fit(X_train, y_train)
            break
        except Exception as e:
            last_err = e
            continue
    else:
        raise RuntimeError(
            f"GPC fit failed under LIN even with jitter; last error:\n{last_err}"
        )

    proba_test = gpc.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= 0.5).astype(int)

    acc   = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, proba_test)
    auc   = roc_auc_score(y_test, proba_test) if np.any(y_test==0) and np.any(y_test==1) else np.nan

    print("="*70)
    print(f"h = {h} | 训练集: {n_train} | 测试集: {test_size} | τ = {tau:.6g}")
    print(f"Optimized kernel (full): {gpc.kernel_}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Brier    : {brier:.4f}")
    print(f"ROC-AUC  : {auc:.4f}" if not np.isnan(auc) else "ROC-AUC : N/A")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

    kernel_str = str(gpc.kernel_).split("+ WhiteKernel")[0].strip()
    kernel_str_wrapped = "Optimized kernel:\n" + "\n".join(textwrap.wrap(kernel_str, width=40))

    return {
        "t_test": t_test,
        "proba_test": proba_test,
        "y_test": y_test,
        "kernel_str": kernel_str_wrapped
    }

results = [run_gpc_for_h(prices, h) for h in H_LIST]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, h, res in zip(axes, H_LIST, results):
    ax.plot(res["t_test"], res["proba_test"], lw=1.5, label="Predicted P(y=1)")
    ax.scatter(res["t_test"], res["y_test"], s=18, alpha=0.6, label="True label (0/1)")
    ax.axhline(0.5, ls="--", label="0.5 threshold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time index")
    ax.set_title(f"h = {h}\n{res['kernel_str']}", fontsize=9)

axes[0].set_ylabel("Probability / Label")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("GPC (LIN): Test-set Probabilities with Optimized Kernel (noise hidden)", y=1.05)
fig.tight_layout()
plt.show

#----------------------------------------------------------------------------------------------

#Figure34
#LIN + RBF 
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * (
        DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
        + RBF(length_scale=3.0, length_scale_bounds=(1e-2, 1e3))
    )
    gpc = GaussianProcessClassifier(kernel=kernel, random_state=seed, max_iter_predict=200)
    gpc.fit(X_train, y_train)

    proba_test = gpc.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= 0.5).astype(int)

    acc   = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, proba_test)
    auc   = roc_auc_score(y_test, proba_test) if np.any(y_test==0) and np.any(y_test==1) else np.nan

    print("="*70)
    print(f"h = {h} | 训练集: {n_train} | 测试集: {test_size} | τ = {tau:.6g}")
    print(f"Optimized kernel: {gpc.kernel_}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Brier    : {brier:.4f}")
    print(f"ROC-AUC  : {auc:.4f}" if not np.isnan(auc) else "ROC-AUC : N/A")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

    return {
        "t_test": t_test,
        "proba_test": proba_test,
        "y_test": y_test,
        "kernel_str": str(gpc.kernel_) 
    }

results = [run_gpc_for_h(prices, h) for h in H_LIST]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, h, res in zip(axes, H_LIST, results):
    ax.plot(res["t_test"], res["proba_test"], lw=1.5, label="Predicted P(y=1)")
    ax.scatter(res["t_test"], res["y_test"], s=18, alpha=0.6, label="True label (0/1)")
    ax.axhline(0.5, ls="--", label="0.5 threshold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time index")

    ax.set_title(f"h = {h}\nOptimized kernel:\n{res['kernel_str']}", fontsize=8)

axes[0].set_ylabel("Probability / Label")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("GPC (LIN + RBF kernel): Test-set Probabilities with Optimized Kernel Parameters", y=1.05)
fig.tight_layout()
plt.show()

#----------------------------------------------------------------

#Figure35
# PER×RBF
    def make_kernel(jitter):
        return (
            ConstantKernel(1.0, (1e-3, 1e3))
            * (
                ExpSineSquared(
                    length_scale=1.0,
                    periodicity=12.0,
                    length_scale_bounds=(1e-2, 1e3),
                    periodicity_bounds=(1e-1, 1e3)
                )
                * RBF(
                    length_scale=1.0,
                    length_scale_bounds=(1e-2, 1e3)
                )
            )
            + WhiteKernel(
                noise_level=jitter,
                noise_level_bounds=(1e-10, 1e-2)
            )
        )

    jitters = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    last_err = None
    for jit in jitters:
        kernel = make_kernel(jit)
        gpc = GaussianProcessClassifier(
            kernel=kernel,
            random_state=seed,
            max_iter_predict=300,
            optimizer="fmin_l_bfgs_b",
            warm_start=False
        )
        try:
            gpc.fit(X_train, y_train)
            break
        except Exception as e:
            last_err = e
            continue
    else:
        raise RuntimeError(
            f"GPC fit failed under PER×RBF even with jitter; last error:\n{last_err}"
        )

    proba_test = gpc.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= 0.5).astype(int)

    acc   = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, proba_test)
    auc   = roc_auc_score(y_test, proba_test) if np.any(y_test==0) and np.any(y_test==1) else np.nan

    print("="*70)
    print(f"h = {h} | 训练集: {n_train} | 测试集: {test_size} | τ = {tau:.6g}")
    print(f"Optimized kernel: {gpc.kernel_}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Brier    : {brier:.4f}")
    print(f"ROC-AUC  : {auc:.4f}" if not np.isnan(auc) else "ROC-AUC : N/A")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

    kernel_str = str(gpc.kernel_).split("+ WhiteKernel")[0].strip()
    kernel_str_wrapped = "\n".join(textwrap.wrap(kernel_str, width=50))

    return {
        "t_test": t_test,
        "proba_test": proba_test,
        "y_test": y_test,
        "kernel_str": kernel_str_wrapped
    }

results = [run_gpc_for_h(prices, h) for h in H_LIST]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, h, res in zip(axes, H_LIST, results):
    ax.plot(res["t_test"], res["proba_test"], lw=1.5, label="Predicted P(y=1)")
    ax.scatter(res["t_test"], res["y_test"], s=18, alpha=0.6, label="True label (0/1)")
    ax.axhline(0.5, ls="--", label="0.5 threshold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time index")

    ax.set_title(f"h = {h}\nOptimised kernel: {res['kernel_str']}", fontsize=10)

axes[0].set_ylabel("Probability / Label")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("GPC (PER × RBF): Test-set Probabilities with Optimized Kernel", y=1.05)
fig.tight_layout()
plt.show()

#------------------------------------------------------------------------------

#Figure36
# RBF + Matern
    def make_kernel(jitter):
        return (
            ConstantKernel(1.0, (1e-3, 1e3))
            * (RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
               + Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=1.5))
            + WhiteKernel(noise_level=jitter, noise_level_bounds=(1e-10, 1e-2))
        )

    jitters = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    last_err = None
    for jit in jitters:
        kernel = make_kernel(jit)
        gpc = GaussianProcessClassifier(
            kernel=kernel,
            random_state=seed,
            max_iter_predict=300,
            optimizer="fmin_l_bfgs_b",
            warm_start=False
        )
        try:
            gpc.fit(X_train, y_train)
            break
        except Exception as e:
            last_err = e
            continue
    else:
        raise RuntimeError(
            f"GPC fit failed under RBF+Matern even with jitter; last error:\n{last_err}"
        )

    proba_test = gpc.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= 0.5).astype(int)

    acc   = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, proba_test)
    auc   = roc_auc_score(y_test, proba_test) if np.any(y_test==0) and np.any(y_test==1) else np.nan

    print("="*70)
    print(f"h = {h} | 训练集: {n_train} | 测试集: {test_size} | τ = {tau:.6g}")
    print(f"Optimized kernel (full): {gpc.kernel_}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Brier    : {brier:.4f}")
    print(f"ROC-AUC  : {auc:.4f}" if not np.isnan(auc) else "ROC-AUC : N/A")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

    kernel_str = str(gpc.kernel_).split("+ WhiteKernel")[0].strip()
    kernel_str_wrapped = "Optimized kernel: " + "\n".join(textwrap.wrap(kernel_str, width=40))

    return {
        "t_test": t_test,
        "proba_test": proba_test,
        "y_test": y_test,
        "kernel_str": kernel_str_wrapped
    }

results = [run_gpc_for_h(prices, h) for h in H_LIST]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, h, res in zip(axes, H_LIST, results):
    ax.plot(res["t_test"], res["proba_test"], lw=1.5, label="Predicted P(y=1)")
    ax.scatter(res["t_test"], res["y_test"], s=18, alpha=0.6, label="True label (0/1)")
    ax.axhline(0.5, ls="--", label="0.5 threshold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time index")
    ax.set_title(f"h = {h}\n{res['kernel_str']}", fontsize=9)

axes[0].set_ylabel("Probability / Label")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("GPC (RBF + Matern): Test-set Probabilities with Optimized Kernel (noise hidden)", y=1.05)
fig.tight_layout()
plt.show()

