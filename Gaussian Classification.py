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
    RBF, ConstantKernel, WhiteKernel
)

# ========= 全局参数 =========
DATA_PATH = "E:/系统默认/桌面/eggprice.csv"
N_LAGS    = 6
TEST_SIZE = 105
TAU_MODE  = "median"   
H_LIST    = [1, 4, 8]

# ========= 读取数据 =========
df = pd.read_csv(DATA_PATH)
if "price" in df.columns:
    prices = df["price"].to_numpy(dtype=float)
else:
    if df.shape[1] < 2:
        raise ValueError("CSV 需包含 'price' 列，或至少两列（价格在第2列）。")
    prices = df.iloc[:, 1].to_numpy(dtype=float)

T = len(prices)

# ========= 拟合与评估 =========
def run_gpc_for_h(prices, h, n_lags=N_LAGS, test_size=TEST_SIZE, tau_mode=TAU_MODE, seed=42):
    if T <= (n_lags + h + test_size - 1):
        raise ValueError(f"h={h} 样本不足。")

    # 构造特征与未来价格
    X_list, p_future_list, t_index = [], [], []
    for t in range(n_lags, T - h):
        X_list.append(prices[t - n_lags:t][::-1])
        p_future_list.append(prices[t + h])
        t_index.append(t)

    X_all = np.vstack(X_list)
    p_future_all = np.asarray(p_future_list)
    t_index = np.asarray(t_index)

    # 时间切分
    n_all   = X_all.shape[0]
    n_train = n_all - test_size
    X_train_raw, X_test_raw = X_all[:n_train], X_all[n_train:]
    p_train_fut, p_test_fut = p_future_all[:n_train], p_future_all[n_train:]
    t_train, t_test = t_index[:n_train], t_index[n_train:]

    # 阈值与标签（仅用训练集确定）
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

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    # === 核函数：RBF + WhiteKernel(jitter) ===
    def make_kernel(jitter):
        return (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
            + WhiteKernel(noise_level=jitter, noise_level_bounds=(1e-10, 1e-2))
        )

    # 逐步增大 jitter 的回退策略
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

    # 预测与评估
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

    # 去掉 + WhiteKernel 部分，并自动换行
    kernel_str = str(gpc.kernel_).split("+ WhiteKernel")[0].strip()
    kernel_str_wrapped = "Optimized kernel:\n" + "\n".join(textwrap.wrap(kernel_str, width=40))

    return {
        "t_test": t_test,
        "proba_test": proba_test,
        "y_test": y_test,
        "kernel_str": kernel_str_wrapped
    }

# ========= 跑 h=1,4,8 =========
results = [run_gpc_for_h(prices, h) for h in H_LIST]

# ========= 绘图 =========
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


# RQ

# In[32]:


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap  # 自动换行

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
    confusion_matrix, classification_report
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    RationalQuadratic, ConstantKernel, WhiteKernel
)

# ========= 全局参数 =========
DATA_PATH = "E:/系统默认/桌面/eggprice.csv"
N_LAGS    = 6
TEST_SIZE = 105
TAU_MODE  = "median"   # "median" | "mean" | 数值
H_LIST    = [1, 4, 8]

# ========= 读取数据 =========
df = pd.read_csv(DATA_PATH)
if "price" in df.columns:
    prices = df["price"].to_numpy(dtype=float)
else:
    if df.shape[1] < 2:
        raise ValueError("CSV 需包含 'price' 列，或至少两列（价格在第2列）。")
    prices = df.iloc[:, 1].to_numpy(dtype=float)

T = len(prices)

# ========= 拟合与评估 =========
def run_gpc_for_h(prices, h, n_lags=N_LAGS, test_size=TEST_SIZE, tau_mode=TAU_MODE, seed=42):
    if T <= (n_lags + h + test_size - 1):
        raise ValueError(f"h={h} 样本不足。")

    # 构造特征与未来价格
    X_list, p_future_list, t_index = [], [], []
    for t in range(n_lags, T - h):
        X_list.append(prices[t - n_lags:t][::-1])
        p_future_list.append(prices[t + h])
        t_index.append(t)

    X_all = np.vstack(X_list)
    p_future_all = np.asarray(p_future_list)
    t_index = np.asarray(t_index)

    # 时间切分
    n_all   = X_all.shape[0]
    n_train = n_all - test_size
    X_train_raw, X_test_raw = X_all[:n_train], X_all[n_train:]
    p_train_fut, p_test_fut = p_future_all[:n_train], p_future_all[n_train:]
    t_train, t_test = t_index[:n_train], t_index[n_train:]

    # 阈值与标签（仅用训练集确定）
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

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    # === 核函数：RQ + WhiteKernel(jitter) ===
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

    # 逐步增大 jitter 的回退策略
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

    # 预测与评估
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

    # 去掉 + WhiteKernel 部分，并自动换行
    kernel_str = str(gpc.kernel_).split("+ WhiteKernel")[0].strip()
    kernel_str_wrapped = "Optimized kernel:\n" + "\n".join(textwrap.wrap(kernel_str, width=70))

    return {
        "t_test": t_test,
        "proba_test": proba_test,
        "y_test": y_test,
        "kernel_str": kernel_str_wrapped
    }

# ========= 跑 h=1,4,8 =========
results = [run_gpc_for_h(prices, h) for h in H_LIST]

# ========= 绘图 =========
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


# Matern

# In[34]:


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap  # 自动换行

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
    confusion_matrix, classification_report
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    Matern, ConstantKernel, WhiteKernel
)

# ========= 全局参数 =========
DATA_PATH = "E:/系统默认/桌面/eggprice.csv"
N_LAGS    = 6
TEST_SIZE = 105
TAU_MODE  = "median"   # "median" | "mean" | 数值
H_LIST    = [1, 4, 8]

# ========= 读取数据 =========
df = pd.read_csv(DATA_PATH)
if "price" in df.columns:
    prices = df["price"].to_numpy(dtype=float)
else:
    if df.shape[1] < 2:
        raise ValueError("CSV 需包含 'price' 列，或至少两列（价格在第2列）。")
    prices = df.iloc[:, 1].to_numpy(dtype=float)

T = len(prices)

# ========= 拟合与评估 =========
def run_gpc_for_h(prices, h, n_lags=N_LAGS, test_size=TEST_SIZE, tau_mode=TAU_MODE, seed=42):
    if T <= (n_lags + h + test_size - 1):
        raise ValueError(f"h={h} 样本不足。")

    # 构造特征与未来价格
    X_list, p_future_list, t_index = [], [], []
    for t in range(n_lags, T - h):
        X_list.append(prices[t - n_lags:t][::-1])
        p_future_list.append(prices[t + h])
        t_index.append(t)

    X_all = np.vstack(X_list)
    p_future_all = np.asarray(p_future_list)
    t_index = np.asarray(t_index)

    # 时间切分
    n_all   = X_all.shape[0]
    n_train = n_all - test_size
    X_train_raw, X_test_raw = X_all[:n_train], X_all[n_train:]
    p_train_fut, p_test_fut = p_future_all[:n_train], p_future_all[n_train:]
    t_train, t_test = t_index[:n_train], t_index[n_train:]

    # 阈值与标签（仅用训练集确定）
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

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    # === 核函数：Matern + WhiteKernel(jitter) ===
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

    # 逐步增大 jitter 的回退策略
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

    # 预测与评估
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

    # 去掉 + WhiteKernel 部分，并自动换行
    kernel_str = str(gpc.kernel_).split("+ WhiteKernel")[0].strip()
    kernel_str_wrapped = "Optimized kernel:\n" + "\n".join(textwrap.wrap(kernel_str, width=60))

    return {
        "t_test": t_test,
        "proba_test": proba_test,
        "y_test": y_test,
        "kernel_str": kernel_str_wrapped
    }

# ========= 跑 h=1,4,8 =========
results = [run_gpc_for_h(prices, h) for h in H_LIST]

# ========= 绘图 =========
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


# In[ ]:


LIN


# In[36]:


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap  # 自动换行

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
    confusion_matrix, classification_report
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    DotProduct, ConstantKernel, WhiteKernel
)

# ========= 全局参数 =========
DATA_PATH = "E:/系统默认/桌面/eggprice.csv"
N_LAGS    = 6
TEST_SIZE = 105
TAU_MODE  = "median"   # "median" | "mean" | 数值
H_LIST    = [1, 4, 8]

# ========= 读取数据 =========
df = pd.read_csv(DATA_PATH)
if "price" in df.columns:
    prices = df["price"].to_numpy(dtype=float)
else:
    if df.shape[1] < 2:
        raise ValueError("CSV 需包含 'price' 列，或至少两列（价格在第2列）。")
    prices = df.iloc[:, 1].to_numpy(dtype=float)

T = len(prices)

# ========= 拟合与评估 =========
def run_gpc_for_h(prices, h, n_lags=N_LAGS, test_size=TEST_SIZE, tau_mode=TAU_MODE, seed=42):
    if T <= (n_lags + h + test_size - 1):
        raise ValueError(f"h={h} 样本不足。")

    # 构造特征与未来价格
    X_list, p_future_list, t_index = [], [], []
    for t in range(n_lags, T - h):
        X_list.append(prices[t - n_lags:t][::-1])
        p_future_list.append(prices[t + h])
        t_index.append(t)

    X_all = np.vstack(X_list)
    p_future_all = np.asarray(p_future_list)
    t_index = np.asarray(t_index)

    # 时间切分
    n_all   = X_all.shape[0]
    n_train = n_all - test_size
    X_train_raw, X_test_raw = X_all[:n_train], X_all[n_train:]
    p_train_fut, p_test_fut = p_future_all[:n_train], p_future_all[n_train:]
    t_train, t_test = t_index[:n_train], t_index[n_train:]

    # 阈值与标签（仅用训练集确定）
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

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    # === 核函数：LIN + WhiteKernel(jitter) ===
    def make_kernel(jitter):
        return (
            ConstantKernel(1.0, (1e-3, 1e3))
            * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
            + WhiteKernel(noise_level=jitter, noise_level_bounds=(1e-10, 1e-2))
        )

    # 逐步增大 jitter 的回退策略
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

    # 预测与评估
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

    # 去掉 + WhiteKernel 部分，并自动换行
    kernel_str = str(gpc.kernel_).split("+ WhiteKernel")[0].strip()
    kernel_str_wrapped = "Optimized kernel:\n" + "\n".join(textwrap.wrap(kernel_str, width=40))

    return {
        "t_test": t_test,
        "proba_test": proba_test,
        "y_test": y_test,
        "kernel_str": kernel_str_wrapped
    }

# ========= 跑 h=1,4,8 =========
results = [run_gpc_for_h(prices, h) for h in H_LIST]

# ========= 绘图 =========
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


# LIN+RBF

# In[2]:


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
    confusion_matrix, classification_report
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, RBF, ConstantKernel

# ========= 全局参数 =========
DATA_PATH = "E:/系统默认/桌面/eggprice.csv"
N_LAGS    = 6
TEST_SIZE = 105
TAU_MODE  = "median"   # "median" | "mean" | 数值
H_LIST    = [1, 4, 8]

# ========= 读取数据 =========
df = pd.read_csv(DATA_PATH)
if "price" in df.columns:
    prices = df["price"].to_numpy(dtype=float)
else:
    if df.shape[1] < 2:
        raise ValueError("CSV 需包含 'price' 列，或至少两列（价格在第2列）。")
    prices = df.iloc[:, 1].to_numpy(dtype=float)

T = len(prices)

# ========= 辅助函数 =========
def run_gpc_for_h(prices, h, n_lags=N_LAGS, test_size=TEST_SIZE, tau_mode=TAU_MODE, seed=42):
    if T <= (n_lags + h + test_size - 1):
        raise ValueError(f"h={h} 样本不足。")

    # 构造 X 与 p_{t+h}
    X_list, p_future_list, t_index = [], [], []
    for t in range(n_lags, T - h):
        X_list.append(prices[t - n_lags:t][::-1])
        p_future_list.append(prices[t + h])
        t_index.append(t)

    X_all = np.vstack(X_list)
    p_future_all = np.asarray(p_future_list)
    t_index = np.asarray(t_index)

    # 时间切分
    n_all   = X_all.shape[0]
    n_train = n_all - test_size
    X_train_raw, X_test_raw = X_all[:n_train], X_all[n_train:]
    p_train_fut, p_test_fut = p_future_all[:n_train], p_future_all[n_train:]
    t_train, t_test = t_index[:n_train], t_index[n_train:]

    # 阈值
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

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    # === GPC：LIN + RBF 组合核 ===
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * (
        DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
        + RBF(length_scale=3.0, length_scale_bounds=(1e-2, 1e3))
    )
    gpc = GaussianProcessClassifier(kernel=kernel, random_state=seed, max_iter_predict=200)
    gpc.fit(X_train, y_train)

    # 预测与评估
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
        "kernel_str": str(gpc.kernel_)  # 保存优化后核函数的完整字符串
    }

# ========= 跑 h=1,4,8 =========
results = [run_gpc_for_h(prices, h) for h in H_LIST]

# ========= 绘图 =========
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, h, res in zip(axes, H_LIST, results):
    ax.plot(res["t_test"], res["proba_test"], lw=1.5, label="Predicted P(y=1)")
    ax.scatter(res["t_test"], res["y_test"], s=18, alpha=0.6, label="True label (0/1)")
    ax.axhline(0.5, ls="--", label="0.5 threshold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time index")

    # 图标题直接显示完整核函数
    ax.set_title(f"h = {h}\nOptimized kernel:\n{res['kernel_str']}", fontsize=8)

axes[0].set_ylabel("Probability / Label")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("GPC (LIN + RBF kernel): Test-set Probabilities with Optimized Kernel Parameters", y=1.05)
fig.tight_layout()
plt.show()


# PER×RBF

# In[21]:


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap  # 自动换行

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
    confusion_matrix, classification_report
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    RBF, ExpSineSquared, ConstantKernel, WhiteKernel
)

# ========= 全局参数 =========
DATA_PATH = "E:/系统默认/桌面/eggprice.csv"
N_LAGS    = 6
TEST_SIZE = 105
TAU_MODE  = "median"   # "median" | "mean" | 数值
H_LIST    = [1, 4, 8]

# ========= 读取数据 =========
df = pd.read_csv(DATA_PATH)
if "price" in df.columns:
    prices = df["price"].to_numpy(dtype=float)
else:
    if df.shape[1] < 2:
        raise ValueError("CSV 需包含 'price' 列，或至少两列（价格在第2列）。")
    prices = df.iloc[:, 1].to_numpy(dtype=float)

T = len(prices)

# ========= 拟合与评估（带稳健回退）=========
def run_gpc_for_h(prices, h, n_lags=N_LAGS, test_size=TEST_SIZE, tau_mode=TAU_MODE, seed=42):
    if T <= (n_lags + h + test_size - 1):
        raise ValueError(f"h={h} 样本不足。")

    # 构造特征与未来价格
    X_list, p_future_list, t_index = [], [], []
    for t in range(n_lags, T - h):
        X_list.append(prices[t - n_lags:t][::-1])
        p_future_list.append(prices[t + h])
        t_index.append(t)

    X_all = np.vstack(X_list)
    p_future_all = np.asarray(p_future_list)
    t_index = np.asarray(t_index)

    # 时间切分
    n_all   = X_all.shape[0]
    n_train = n_all - test_size
    X_train_raw, X_test_raw = X_all[:n_train], X_all[n_train:]
    p_train_fut, p_test_fut = p_future_all[:n_train], p_future_all[n_train:]
    t_train, t_test = t_index[:n_train], t_index[n_train:]

    # 阈值与标签（仅用训练集确定）
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

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    # === 初始核： PER × RBF，加上 WhiteKernel 作为 jitter ===
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

    # 逐步增大 jitter 的回退策略
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

    # 预测与评估
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

    # 去掉 + WhiteKernel 部分，并自动换行
    kernel_str = str(gpc.kernel_).split("+ WhiteKernel")[0].strip()
    kernel_str_wrapped = "\n".join(textwrap.wrap(kernel_str, width=50))

    return {
        "t_test": t_test,
        "proba_test": proba_test,
        "y_test": y_test,
        "kernel_str": kernel_str_wrapped
    }

# ========= 跑 h=1,4,8 =========
results = [run_gpc_for_h(prices, h) for h in H_LIST]

# ========= 绘图 =========
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, h, res in zip(axes, H_LIST, results):
    ax.plot(res["t_test"], res["proba_test"], lw=1.5, label="Predicted P(y=1)")
    ax.scatter(res["t_test"], res["y_test"], s=18, alpha=0.6, label="True label (0/1)")
    ax.axhline(0.5, ls="--", label="0.5 threshold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time index")

    # 使用自动换行后的标题
    ax.set_title(f"h = {h}\nOptimised kernel: {res['kernel_str']}", fontsize=10)

axes[0].set_ylabel("Probability / Label")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("GPC (PER × RBF): Test-set Probabilities with Optimized Kernel", y=1.05)
fig.tight_layout()
plt.show()


# In[ ]:





# RBF + Matern

# In[27]:


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap  # 自动换行

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
    confusion_matrix, classification_report
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    RBF, Matern, ConstantKernel, WhiteKernel
)

# ========= 全局参数 =========
DATA_PATH = "E:/系统默认/桌面/eggprice.csv"
N_LAGS    = 6
TEST_SIZE = 105
TAU_MODE  = "median"   # "median" | "mean" | 数值
H_LIST    = [1, 4, 8]

# ========= 读取数据 =========
df = pd.read_csv(DATA_PATH)
if "price" in df.columns:
    prices = df["price"].to_numpy(dtype=float)
else:
    if df.shape[1] < 2:
        raise ValueError("CSV 需包含 'price' 列，或至少两列（价格在第2列）。")
    prices = df.iloc[:, 1].to_numpy(dtype=float)

T = len(prices)

# ========= 拟合与评估（带稳健回退）=========
def run_gpc_for_h(prices, h, n_lags=N_LAGS, test_size=TEST_SIZE, tau_mode=TAU_MODE, seed=42):
    if T <= (n_lags + h + test_size - 1):
        raise ValueError(f"h={h} 样本不足。")

    # 构造特征与未来价格
    X_list, p_future_list, t_index = [], [], []
    for t in range(n_lags, T - h):
        X_list.append(prices[t - n_lags:t][::-1])
        p_future_list.append(prices[t + h])
        t_index.append(t)

    X_all = np.vstack(X_list)
    p_future_all = np.asarray(p_future_list)
    t_index = np.asarray(t_index)

    # 时间切分
    n_all   = X_all.shape[0]
    n_train = n_all - test_size
    X_train_raw, X_test_raw = X_all[:n_train], X_all[n_train:]
    p_train_fut, p_test_fut = p_future_all[:n_train], p_future_all[n_train:]
    t_train, t_test = t_index[:n_train], t_index[n_train:]

    # 阈值与标签（仅用训练集确定）
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

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    # === 初始核： RBF + Matern，加 WhiteKernel 作为 jitter ===
    def make_kernel(jitter):
        return (
            ConstantKernel(1.0, (1e-3, 1e3))
            * (RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
               + Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=1.5))
            + WhiteKernel(noise_level=jitter, noise_level_bounds=(1e-10, 1e-2))
        )

    # 逐步增大 jitter 的回退策略
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

    # 预测与评估
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

    # —— 图标题使用的字符串：隐藏 WhiteKernel，并自动换行 —— #
    kernel_str = str(gpc.kernel_).split("+ WhiteKernel")[0].strip()
    kernel_str_wrapped = "Optimized kernel: " + "\n".join(textwrap.wrap(kernel_str, width=40))

    return {
        "t_test": t_test,
        "proba_test": proba_test,
        "y_test": y_test,
        "kernel_str": kernel_str_wrapped
    }

# ========= 跑 h=1,4,8 =========
results = [run_gpc_for_h(prices, h) for h in H_LIST]

# ========= 绘图 =========
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


# In[ ]:





# In[ ]:





# In[39]:


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegressionCV

# =============== 全局参数（按需改动） ===============
DATA_PATH = "E:/系统默认/桌面/eggprice.csv"
N_LAGS    = 6                 # 滑动窗口长度
TEST_SIZE = 105               # 测试集长度（按时间后切）
TAU_MODE  = "median"          # "median" | "mean" | 数值(如 7.20)
H_LIST    = [1, 4, 8]         # 预测步长集合
RANDOM_SEED = 42

# =============== 读取价格序列 ===============
df = pd.read_csv(DATA_PATH)
if "price" in df.columns:
    prices = df["price"].to_numpy(dtype=float)
else:
    if df.shape[1] < 2:
        raise ValueError("CSV 需包含 'price' 列，或至少两列（价格在第2列）。")
    prices = df.iloc[:, 1].to_numpy(dtype=float)

T = len(prices)

# =============== 核心函数：给定 h 训练与评估 ===============
def run_logreg_for_h(prices, h, n_lags=N_LAGS, test_size=TEST_SIZE, tau_mode=TAU_MODE, seed=RANDOM_SEED):
    if T <= (n_lags + h + test_size - 1):
        raise ValueError(f"h={h} 样本不足：需要 T > n_lags + h + test_size - 1")

    # 1) 构造特征 X_t = [p_{t}, p_{t-1}, ..., p_{t-n_lags+1}], 目标为 p_{t+h}
    X_list, p_future_list, t_index = [], [], []
    for t in range(n_lags, T - h):
        X_list.append(prices[t - n_lags:t][::-1])  # 近→远
        p_future_list.append(prices[t + h])
        t_index.append(t)

    X_all = np.vstack(X_list)
    p_future_all = np.asarray(p_future_list)
    t_index = np.asarray(t_index)

    # 2) 时间切分（后段为测试集）
    n_all   = X_all.shape[0]
    n_train = n_all - test_size
    X_train_raw, X_test_raw = X_all[:n_train], X_all[n_train:]
    p_train_fut, p_test_fut = p_future_all[:n_train], p_future_all[n_train:]
    t_train, t_test = t_index[:n_train], t_index[n_train:]

    # 3) 用训练集确定阈值 τ，并生成标签
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

    # 4) 标准化（仅用训练集参数）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    # 5) 逻辑回归 + 交叉验证选择 C（评分：ROC-AUC）
    #    若关注校准，也可把 scoring 改为 "neg_brier_score"
    logit_cv = LogisticRegressionCV(
        Cs=np.logspace(-3, 3, 25),
        cv=5,
        penalty="l2",
        solver="lbfgs",
        scoring="roc_auc",
        max_iter=2000,
        n_jobs=None,
        refit=True,
        random_state=seed
    )
    logit_cv.fit(X_train, y_train)

    # 6) 预测与评估
    proba_test = logit_cv.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= 0.5).astype(int)

    acc   = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, proba_test)
    auc   = roc_auc_score(y_test, proba_test) if (y_test.min()==0 and y_test.max()==1) else np.nan

    # 打印关键信息
    print("=" * 72)
    print(f"h = {h} | 训练集: {n_train} | 测试集: {test_size} | τ = {tau:.6g}")
    print(f"Best C (mean across classes): {float(logit_cv.C_[0]):.6g}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Brier    : {brier:.4f}")
    print(f"ROC-AUC  : {auc:.4f}" if not np.isnan(auc) else "ROC-AUC : N/A")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

    title_str = f"Logistic Regression (best C={float(logit_cv.C_[0]):.3g})"

    return {
        "h": h,
        "tau": tau,
        "t_test": t_test,
        "proba_test": proba_test,
        "y_test": y_test,
        "acc": acc, "auc": auc, "brier": brier,
        "title": title_str
    }

# =============== 跑 h=1,4,8 并作图 ===============
results = [run_logreg_for_h(prices, h) for h in H_LIST]

# 概率三联图
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, res in zip(axes, results):
    ax.plot(res["t_test"], res["proba_test"], lw=1.5, label="Predicted P(y=1)")
    ax.scatter(res["t_test"], res["y_test"], s=18, alpha=0.65, label="True label (0/1)")
    ax.axhline(0.5, ls="--", label="0.5 threshold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time index")
    ax.set_title(f"h = {res['h']}\n{res['title']}", fontsize=9)

axes[0].set_ylabel("Probability / Label")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("Logistic Regression: Test-set Probabilities", y=1.05)
fig.tight_layout()
plt.show()

# =============== 结果表：便于论文/导出 ===============
table = pd.DataFrame({
    "h": [r["h"] for r in results],
    "Accuracy": [r["acc"] for r in results],
    "ROC-AUC": [r["auc"] for r in results],
    "Brier": [r["brier"] for r in results],
    "tau": [r["tau"] for r in results]
})
print("\nSummary table:")
print(table.to_string(index=False))

# 保存为 CSV（可选）
table.to_csv("logreg_results_summary.csv", index=False)
print("\nSaved: logreg_results_summary.csv")


# In[ ]:




