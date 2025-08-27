import numpy as np, matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.stats import multivariate_normal
from sklearn.gaussian_process.kernels import (
    RBF, RationalQuadratic, Matern, ExpSineSquared,
    DotProduct, WhiteKernel, ConstantKernel
)

# --------------------------------------------------------------------------

#Figure 6
#  Settings 
rng = default_rng(33)
n_samples = 5

x_left = np.linspace(0.0, 5.0, 300)      # Left column: domain for GP prior samples (1D)
r_sym = np.linspace(-5.0, 5.0, 400)      # for RBF/RQ (symmetric about 0)   #Right column
r_pos = np.linspace(0.0,  5.0, 400)      # for Matérn (monotone decay)

# Kernel hyperparameters (as requested)
ell = 1.0         # length-scale
alpha = 2.0       # RQ shape
nu = 1.5          # Matérn smoothness
sigma2 = 1.0      # amplitude σ^2

# Visual style
left_ylim  = (-3.8, 2.2)
right_ylim = (0.0, 1.05)
lw = 2.2

fig, axes = plt.subplots(3, 2, figsize=(10, 6),
                         gridspec_kw={"wspace": 0.35, "hspace": 0.65})

def style_axes(ax, left=True):
    """Thicker spines and clean look."""
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    ax.tick_params(axis='both', labelsize=9, width=1.2, length=3)
    ax.set_ylim(*(left_ylim if left else right_ylim))

def gp_prior_samples(kernel, X, n_funcs, rng):
    """Draw samples from a zero-mean GP prior with kernel k and 1D inputs X."""
    K = kernel(X[:, None])
    K = K + 1e-8 * np.eye(len(X))         # jitter for numerical stability
    S = rng.multivariate_normal(np.zeros(len(X)), K, size=n_funcs).T
    return S  # shape (len(X), n_funcs)

# Closed-form kernel profiles k(r) with σ^2=1, ℓ=1 by default
def k_rbf(r, ell=1.0, sigma2=1.0):
    return sigma2 * np.exp(-(r**2) / (2.0 * ell**2))

def k_matern_15(r, ell=1.0, sigma2=1.0):
    # ν = 3/2: k(r) = σ^2 (1 + √3 r/ℓ) exp(-√3 r/ℓ), r >= 0
    a = np.sqrt(3.0) * (np.abs(r) / ell)
    return sigma2 * (1.0 + a) * np.exp(-a)

def k_rq(r, ell=1.0, alpha=2.0, sigma2=1.0):
    # k(r) = σ^2 (1 + r^2/(2 α ℓ^2))^{-α}
    return sigma2 * (1.0 + (r**2) / (2.0 * alpha * ell**2)) ** (-alpha)

# Kernels for left column sampling
# Use sklearn for sampling (same hyperparameters), amplitude = 1
kernels_for_sampling = [
    ("RBF",    RBF(length_scale=ell)),
    ("Matérn", Matern(length_scale=ell, nu=nu)),
    ("RQ",     RationalQuadratic(length_scale=ell, alpha=alpha)),
]

# Plotting 
for row, (name, k_skl) in enumerate(kernels_for_sampling):
    # Left: GP prior samples
    axL = axes[row, 0]
    Y = gp_prior_samples(k_skl, x_left, n_samples, rng)
    for i in range(n_samples):
        axL.plot(x_left, Y[:, i], linewidth=lw)
    axL.set_xlim(x_left.min(), x_left.max())
    style_axes(axL, left=True)
    axL.set_ylabel(name, fontsize=11, rotation=90, labelpad=8)

    # Right: analytic kernel profile k(r)
    axR = axes[row, 1]
    if name == "RBF":
        r = r_sym
        axR.plot(r, k_rbf(r, ell=ell, sigma2=sigma2), linewidth=lw)
        axR.set_xlim(r.min(), r.max())
    elif name == "Matérn":
        r = r_pos  # monotone decay for r >= 0
        axR.plot(r, k_matern_15(r, ell=ell, sigma2=sigma2), linewidth=lw)
        axR.set_xlim(r.min(), r.max())
    else:  # RQ
        r = r_sym
        axR.plot(r, k_rq(r, ell=ell, alpha=alpha, sigma2=sigma2), linewidth=lw)
        axR.set_xlim(r.min(), r.max())

    style_axes(axR, left=False)
    axR.set_ylabel(r"$k(r)$", fontsize=11, rotation=90, labelpad=8)
    axR.yaxis.set_label_coords(-0.075, 0.5)

# Clean x-labels for a minimal look
for ax in axes.ravel():
    ax.set_xlabel("")

plt.savefig("kernel_prior_and_profiles_correct_profiles.png", dpi=300, bbox_inches="tight")
plt.show()

#-----------------------------------------------------------------------------

#Figure 7
rng = default_rng(33)         
n_samples = 5

x_left = np.linspace(0.0, 5.0, 300)
r = np.linspace(-5.0, 5.0, 400)

ell = 1.0      # length-scale ℓ for PER
p   = 3.0      # period p for PER (shorter -> multiple cycles on [-5,5])
sigma2 = 1.0   # amplitude σ^2 (implicit 1 in sklearn)
sigma0 = 0.0   # DotProduct bias; 0 gives pure linear kernel

# Visual style
lw = 2.2

fig, axes = plt.subplots(2, 2, figsize=(9, 5.2),
                         gridspec_kw={"wspace": 0.35, "hspace": 0.60})

def style_axes(ax, left_ylim=None, right_ylim=None, left=True):
    """Thicker spines and clean look."""
    for s in ax.spines.values():
        s.set_linewidth(1.4)
    ax.tick_params(axis='both', labelsize=9, width=1.2, length=3)
    if left and left_ylim is not None:
        ax.set_ylim(*left_ylim)
    if (not left) and right_ylim is not None:
        ax.set_ylim(*right_ylim)

def gp_prior_samples(kernel, X, n_funcs, rng):
    """Draw samples from a zero-mean GP prior with kernel k and inputs X (1D)."""
    K = kernel(X[:, None])
    K = K + 1e-8 * np.eye(len(X))         # jitter for numerical stability
    S = rng.multivariate_normal(np.zeros(len(X)), K, size=n_funcs).T
    return S  # shape (len(X), n_funcs)

# Analytic kernel profiles on r ∈ [-5,5]
def k_per(r, ell=1.0, p=3.0, sigma2=1.0):
    # k_PER(r) = σ^2 * exp( - 2 sin^2(π r / p) / ℓ^2 )
    return sigma2 * np.exp(-2.0 * np.sin(np.pi * r / p)**2 / ell**2)

def k_lin(r):
    # Linear kernel (dot-product style)
    return r  # slope 1 for visual clarity

kernels_for_sampling = [
    ("PER", ExpSineSquared(length_scale=ell, periodicity=p)),
    ("LIN", DotProduct(sigma_0=sigma0)),   # pure linear (no bias)
]

# Plotting
for row, (name, k_skl) in enumerate(kernels_for_sampling):
    # Left: GP prior samples
    axL = axes[row, 0]
    Y = gp_prior_samples(k_skl, x_left, n_samples, rng)
    for i in range(n_samples):
        axL.plot(x_left, Y[:, i], linewidth=lw)
    axL.set_xlim(x_left.min(), x_left.max())
    # set comparable y-ranges (tune if needed)
    axL_ylim = (-2.8, 2.8) if name == "PER" else (-3.2, 3.2)
    style_axes(axL, left_ylim=axL_ylim, left=True)
    axL.set_ylabel(name, fontsize=11, rotation=90, labelpad=8)

    # Right: kernel profile on r ∈ [-5,5]
    axR = axes[row, 1]
    if name == "PER":
        axR.plot(r, k_per(r, ell=ell, p=p, sigma2=sigma2), linewidth=lw)
        axR.set_xlim(r.min(), r.max())
        style_axes(axR, right_ylim=(0.0, 1.05), left=False)
    else:  # LIN
        axR.plot(r, k_lin(r), linewidth=lw)
        axR.set_xlim(r.min(), r.max())
        # allow positive and negative values to show a full straight line
        ymin = min(k_lin(r))
        ymax = max(k_lin(r))
        style_axes(axR, right_ylim=(ymin-0.5, ymax+0.5), left=False)

    axR.set_ylabel(r"$k(r)$", fontsize=11, rotation=90, labelpad=8)
    axR.yaxis.set_label_coords(-0.075, 0.5)
    axR.margins(x=0.03)  # slight padding so the curve isn't glued to the y-axis

# Remove x-label text (cleaner look, like the paper)
for ax in axes.ravel():
    ax.set_xlabel("")

plt.savefig("per_lin_like_paper.png", dpi=300, bbox_inches="tight")
plt.show()

#-----------------------------------------------------------------------------------------------

#Figure 8
np.random.seed(42)                   # Set random seed

# RBF
def calcSigma(X1, X2, theta):
    v1 = theta[0]  # signal variance
    v2 = theta[1]  # noise variance
    alpha = theta[2]  # alpha = 2 for RBF
    r = theta[3]  # length-scale

    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(1, -1)
    dist = np.abs(X1 - X2)
    cov = v1 * np.exp(-(dist / r) ** alpha)
    if X1.shape == X2.T.shape:
        cov += v2 * np.eye(X1.shape[0])
    return cov

def true_function(x):
    return np.log(x + 1) + np.cos(3 * np.pi * x)


# Data generation
x_all = np.sort(np.random.uniform(1, 15, n_total))
y_true = true_function(x_all)
y_obs = y_true + np.random.normal(0, 1, size=n_total)

# Split into training (1,12) and testing (12,15)
train_idx = x_all < 12
x_train = x_all[train_idx]
y_train = y_obs[train_idx]
x_test = x_all[~train_idx]
y_test = y_obs[~train_idx]

# Dense prediction points
x_star = np.linspace(1, 15, 200)

# theta = [v1, v2, alpha, r]
theta = [1.0, 0.1, 2.0, 0.5]  # RBF kernel

# Prior sampling
Sigma_prior = calcSigma(x_star, x_star, theta)
prior_samples = multivariate_normal.rvs(
    mean=np.zeros(len(x_star)),
    cov=Sigma_prior + 1e-6 * np.eye(len(x_star)),
    size=5
)

## Posterior mean and Posterior covariance
K_xx = calcSigma(x_train, x_train, theta)
K_xxs = calcSigma(x_train, x_star, theta)
K_xsx = K_xxs.T
K_xsxs = calcSigma(x_star, x_star, theta)

sigma_n = np.sqrt(theta[1])

K_inv = np.linalg.inv(K_xx + sigma_n**2 * np.eye(len(x_train)))

# Posterior mean
f_star_mean = K_xsx @ K_inv @ y_train
# Posterior covariance
cov_f_star = K_xsxs - K_xsx @ K_inv @ K_xxs

# POSTERIOR sampling
posterior_samples = multivariate_normal.rvs(
    mean=f_star_mean,
    cov=cov_f_star + 1e-6 * np.eye(len(x_star)),
    size=5
)

# Confidence interval 95%
conf_upper = f_star_mean + 1.96 * np.sqrt(np.diag(cov_f_star))
conf_lower = f_star_mean - 1.96 * np.sqrt(np.diag(cov_f_star))

# Plotting
plt.figure(figsize=(18, 4))

# panel 1
plt.subplot(1, 4, 1)
plt.scatter(x_train, y_train, color='blue', label='Train')
plt.scatter(x_test, y_test, color='red', label='Test')
plt.title('Noisy Observations')
plt.xlabel('x')
plt.ylabel('y')
plt.axvline(12, color='gray', linestyle='--')
plt.legend()
plt.grid(True)

# panel 2
plt.subplot(1, 4, 2)
for i in range(5):
    plt.plot(x_star, prior_samples[i], color='orange')
plt.title('GP PRIOR Samples (RBF)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

# panel 3
plt.subplot(1, 4, 3)
for i in range(5):
    plt.plot(x_star, posterior_samples[i], color='green')
plt.title('GP POSTERIOR Samples (RBF)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

# panel 4
plt.subplot(1, 4, 4)
plt.plot(x_star, f_star_mean, color='blue', label='Posterior Mean')
plt.fill_between(x_star, conf_lower, conf_upper, color='gray', alpha=0.4, label='95% CI')
plt.scatter(x_train, y_train, color='blue', s=10)
plt.scatter(x_test, y_test, color='red', s=10)
plt.title('Posterior Mean + 95% CI (RBF Kernel)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------

#Figure 9
np.random.seed(42)

def calcSigma_RQ(X1, X2, theta):
    v1 = theta[0]  # 信号方差
    v2 = theta[1]  # 噪声方差
    alpha = theta[2]  # shape parameter
    r = theta[3]  # length scale

    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(1, -1)
    sqdist = (X1 - X2) ** 2
    cov = v1 * (1 + sqdist / (2 * alpha * r**2)) ** (-alpha)
    if X1.shape == X2.T.shape:
        cov += v2 * np.eye(len(X1))
    return cov

def true_function(x):
    return np.log(x + 1) + np.cos(3 * np.pi * x)

n_total = 40
x_all = np.sort(np.random.uniform(1, 15, n_total))
y_true = true_function(x_all)
y_obs = y_true + np.random.normal(0, 1, size=n_total)

train_idx = x_all < 12
x_train = x_all[train_idx]
y_train = y_obs[train_idx]
x_test = x_all[~train_idx]
y_test = y_obs[~train_idx]

x_star = np.linspace(1, 15, 200)
theta = [1.0, 0.1, 10.0, 0.5]  # Rational Quadratic kernel parameters

Sigma_prior = calcSigma_RQ(x_star, x_star, theta)
prior_samples = multivariate_normal.rvs(
    mean=np.zeros(len(x_star)),
    cov=Sigma_prior + 1e-6 * np.eye(len(x_star)),
    size=5
)

K_xx = calcSigma_RQ(x_train, x_train, theta)
K_xxs = calcSigma_RQ(x_train, x_star, theta)
K_xsx = K_xxs.T
K_xsxs = calcSigma_RQ(x_star, x_star, theta)

sigma_n = np.sqrt(theta[1])
K_inv = np.linalg.inv(K_xx + sigma_n**2 * np.eye(len(x_train)))

f_star_mean = K_xsx @ K_inv @ y_train
cov_f_star = K_xsxs - K_xsx @ K_inv @ K_xxs

posterior_samples = multivariate_normal.rvs(
    mean=f_star_mean,
    cov=cov_f_star + 1e-6 * np.eye(len(x_star)),
    size=5
)

conf_upper = f_star_mean + 1.96 * np.sqrt(np.diag(cov_f_star))
conf_lower = f_star_mean - 1.96 * np.sqrt(np.diag(cov_f_star))

plt.figure(figsize=(18, 4))

plt.subplot(1, 4, 1)
plt.scatter(x_train, y_train, color='blue', label='Train')
plt.scatter(x_test, y_test, color='red', label='Test')
plt.title('Noisy Observations')
plt.xlabel('x')
plt.ylabel('y')
plt.axvline(12, color='gray', linestyle='--')
plt.legend()
plt.grid(True)

plt.subplot(1, 4, 2)
for i in range(5):
    plt.plot(x_star, prior_samples[i], color='orange')
plt.title('GP PRIOR Samples (RQ)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

plt.subplot(1, 4, 3)
for i in range(5):
    plt.plot(x_star, posterior_samples[i], color='green')
plt.title('GP POSTERIOR Samples (RQ)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

plt.subplot(1, 4, 4)
plt.plot(x_star, f_star_mean, color='blue', label='Posterior Mean')
plt.fill_between(x_star, conf_lower, conf_upper, color='gray', alpha=0.4, label='95% CI')
plt.scatter(x_train, y_train, color='blue', s=10)
plt.scatter(x_test, y_test, color='red', s=10)
plt.title('Posterior Mean + 95% CI (RQ Kernel)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#-----------------------------------------------------------------

#Figure 10
np.random.seed(42)

def calcSigma_Matern(X1, X2, theta):
    v1, v2, l = theta
    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(1, -1)
    dists = np.abs(X1 - X2)
    sqrt3_d = np.sqrt(3) * dists / l
    cov = v1 * (1 + sqrt3_d) * np.exp(-sqrt3_d)
    if X1.shape == X2.T.shape:
        cov += v2 * np.eye(len(X1))
    return cov

def true_function(x):
    return np.log(x + 1) + np.cos(3 * np.pi * x)

n_total = 40
x_all = np.sort(np.random.uniform(1, 15, n_total))
y_true = true_function(x_all)
y_obs = y_true + np.random.normal(0, 1, size=n_total)

train_idx = x_all < 12
x_train = x_all[train_idx]
y_train = y_obs[train_idx]
x_test = x_all[~train_idx]
y_test = y_obs[~train_idx]
x_star = np.linspace(1, 15, 200)

theta = [1.0, 0.1, 0.5]  # Matern kernel parameters

Sigma_prior = calcSigma_Matern(x_star, x_star, theta)
prior_samples = multivariate_normal.rvs(
    mean=np.zeros(len(x_star)), cov=Sigma_prior + 1e-6 * np.eye(len(x_star)), size=5
)

K_xx = calcSigma_Matern(x_train, x_train, theta)
K_xxs = calcSigma_Matern(x_train, x_star, theta)
K_xsx = K_xxs.T
K_xsxs = calcSigma_Matern(x_star, x_star, theta)
sigma_n = np.sqrt(theta[1])
K_inv = np.linalg.inv(K_xx + sigma_n**2 * np.eye(len(x_train)))

f_star_mean = K_xsx @ K_inv @ y_train
cov_f_star = K_xsxs - K_xsx @ K_inv @ K_xxs

posterior_samples = multivariate_normal.rvs(
    mean=f_star_mean, cov=cov_f_star + 1e-6 * np.eye(len(x_star)), size=5
)
conf_upper = f_star_mean + 1.96 * np.sqrt(np.diag(cov_f_star))
conf_lower = f_star_mean - 1.96 * np.sqrt(np.diag(cov_f_star))

plt.figure(figsize=(18, 4))
plt.subplot(1, 4, 1)
plt.scatter(x_train, y_train, color='blue', label='Train')
plt.scatter(x_test, y_test, color='red', label='Test')
plt.title('Noisy Observations')
plt.xlabel('x')
plt.ylabel('y')
plt.axvline(12, color='gray', linestyle='--')
plt.legend(); plt.grid(True)

plt.subplot(1, 4, 2)
for i in range(5):
    plt.plot(x_star, prior_samples[i], color='orange')
plt.title('GP PRIOR Samples (Matern)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

plt.subplot(1, 4, 3)
for i in range(5):
    plt.plot(x_star, posterior_samples[i], color='green')
plt.title('GP POSTERIOR Samples (Matern)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

plt.subplot(1, 4, 4)
plt.plot(x_star, f_star_mean, color='blue', label='Posterior Mean')
plt.fill_between(x_star, conf_lower, conf_upper, color='gray', alpha=0.4, label='95% CI')
plt.scatter(x_train, y_train, color='blue', s=10)
plt.scatter(x_test, y_test, color='red', s=10)
plt.title('Posterior Mean + 95% CI (Matern Kernel)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------

#Figure11
np.random.seed(42)

# PER kernel
def calcSigma_PER(X1, X2, theta):
    v1, v2, l, p = theta
    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(1, -1)
    sin_sq = np.sin(np.pi * (X1 - X2) / p) ** 2
    cov = v1 * np.exp(-2 * sin_sq / l**2)
    if X1.shape == X2.T.shape:
        cov += v2 * np.eye(len(X1))
    return cov

def true_function(x):
    return np.log(x + 1) + np.cos(3 * np.pi * x)

n_total = 40
x_all = np.sort(np.random.uniform(1, 15, n_total))
y_true = true_function(x_all)
y_obs = y_true + np.random.normal(0, 1, size=n_total)

train_idx = x_all < 12
x_train = x_all[train_idx]
y_train = y_obs[train_idx]
x_test = x_all[~train_idx]
y_test = y_obs[~train_idx]
x_star = np.linspace(1, 15, 200)

theta = [1.0, 0.1, 0.5, 50]  # Periodic kernel parameters
Sigma_prior = calcSigma_PER(x_star, x_star, theta)
prior_samples = multivariate_normal.rvs(
    mean=np.zeros(len(x_star)), cov=Sigma_prior + 1e-6 * np.eye(len(x_star)), size=5
)

K_xx = calcSigma_PER(x_train, x_train, theta)
K_xxs = calcSigma_PER(x_train, x_star, theta)
K_xsx = K_xxs.T
K_xsxs = calcSigma_PER(x_star, x_star, theta)
sigma_n = np.sqrt(theta[1])
K_inv = np.linalg.inv(K_xx + sigma_n**2 * np.eye(len(x_train)))

f_star_mean = K_xsx @ K_inv @ y_train
cov_f_star = K_xsxs - K_xsx @ K_inv @ K_xxs

posterior_samples = multivariate_normal.rvs(
    mean=f_star_mean, cov=cov_f_star + 1e-6 * np.eye(len(x_star)), size=5
)
conf_upper = f_star_mean + 1.96 * np.sqrt(np.diag(cov_f_star))
conf_lower = f_star_mean - 1.96 * np.sqrt(np.diag(cov_f_star))

plt.figure(figsize=(18, 4))
plt.subplot(1, 4, 1)
plt.scatter(x_train, y_train, color='blue', label='Train')
plt.scatter(x_test, y_test, color='red', label='Test')
plt.title('Noisy Observations')
plt.xlabel('x')
plt.ylabel('y')
plt.axvline(12, color='gray', linestyle='--')
plt.legend(); plt.grid(True)

plt.subplot(1, 4, 2)
for i in range(5):
    plt.plot(x_star, prior_samples[i], color='orange')
plt.title('GP PRIOR Samples (PER)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

plt.subplot(1, 4, 3)
for i in range(5):
    plt.plot(x_star, posterior_samples[i], color='green')
plt.title('GP POSTERIOR Samples (PER)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

plt.subplot(1, 4, 4)
plt.plot(x_star, f_star_mean, color='blue', label='Posterior Mean')
plt.fill_between(x_star, conf_lower, conf_upper, color='gray', alpha=0.4, label='95% CI')
plt.scatter(x_train, y_train, color='blue', s=10)
plt.scatter(x_test, y_test, color='red', s=10)
plt.title('Posterior Mean + 95% CI (PER Kernel)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()

#-------------------------------------------------------------

#Figure 12
np.random.seed(42)

def calcSigma_LIN(X1, X2, theta):
    v1, v2, c = theta
    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(1, -1)
    cov = v1 * (X1 - c) @ (X2 - c)
    if X1.shape == X2.T.shape:
        cov += v2 * np.eye(len(X1))
    return cov

def true_function(x):
    return np.log(x + 1) + np.cos(3 * np.pi * x)

n_total = 40
x_all = np.sort(np.random.uniform(1, 15, n_total))
y_true = true_function(x_all)
y_obs = y_true + np.random.normal(0, 1, size=n_total)

train_idx = x_all < 12
x_train = x_all[train_idx]
y_train = y_obs[train_idx]
x_test = x_all[~train_idx]
y_test = y_obs[~train_idx]
x_star = np.linspace(1, 15, 200)

theta = [0.023, 0.1, 6.0]  # Linear kernel parameters
Sigma_prior = calcSigma_LIN(x_star, x_star, theta)
prior_samples = multivariate_normal.rvs(
    mean=np.zeros(len(x_star)), cov=Sigma_prior + 1e-6 * np.eye(len(x_star)), size=5
)

K_xx = calcSigma_LIN(x_train, x_train, theta)
K_xxs = calcSigma_LIN(x_train, x_star, theta)
K_xsx = K_xxs.T
K_xsxs = calcSigma_LIN(x_star, x_star, theta)
sigma_n = np.sqrt(theta[1])
K_inv = np.linalg.inv(K_xx + sigma_n**2 * np.eye(len(x_train)))

f_star_mean = K_xsx @ K_inv @ y_train
cov_f_star = K_xsxs - K_xsx @ K_inv @ K_xxs

posterior_samples = multivariate_normal.rvs(
    mean=f_star_mean, cov=cov_f_star + 1e-6 * np.eye(len(x_star)), size=5
)
conf_upper = f_star_mean + 1.96 * np.sqrt(np.diag(cov_f_star))
conf_lower = f_star_mean - 1.96 * np.sqrt(np.diag(cov_f_star))

plt.figure(figsize=(18, 4))
plt.subplot(1, 4, 1)
plt.scatter(x_train, y_train, color='blue', label='Train')
plt.scatter(x_test, y_test, color='red', label='Test')
plt.title('Noisy Observations')
plt.xlabel('x')
plt.ylabel('y')
plt.axvline(12, color='gray', linestyle='--')
plt.legend(); plt.grid(True)

plt.subplot(1, 4, 2)
for i in range(5):
    plt.plot(x_star, prior_samples[i], color='orange')
plt.title('GP PRIOR Samples (LIN)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

plt.subplot(1, 4, 3)
for i in range(5):
    plt.plot(x_star, posterior_samples[i], color='green')
plt.title('GP POSTERIOR Samples (LIN)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

plt.subplot(1, 4, 4)
plt.plot(x_star, f_star_mean, color='blue', label='Posterior Mean')
plt.fill_between(x_star, conf_lower, conf_upper, color='gray', alpha=0.4, label='95% CI')
plt.scatter(x_train, y_train, color='blue', s=10)
plt.scatter(x_test, y_test, color='red', s=10)
plt.title('Posterior Mean + 95% CI (LIN Kernel)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()

#---------------------------------------------------------------

#Figure 13
np.random.seed(42)

# RQ Covariance function 
def calcSigma_RQ(X1, X2, theta):
    v1 = theta[0]  # signal variance
    v2 = theta[1]  # noise variance
    alpha = theta[2]  # shape parameter
    r = theta[3]  # length scale

    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(1, -1)
    sqdist = (X1 - X2) ** 2
    cov = v1 * (1 + sqdist / (2 * alpha * r**2)) ** (-alpha)
    if X1.shape == X2.T.shape:
        cov += v2 * np.eye(len(X1))
    return cov

def true_function(x):
    return np.log(x + 1) + np.cos(3 * np.pi * x)

n_total = 40
x_all = np.sort(np.random.uniform(1, 15, n_total))
y_true = true_function(x_all)
y_obs = y_true + np.random.normal(0, 1, size=n_total)

train_idx = x_all < 12
x_train = x_all[train_idx]
y_train = y_obs[train_idx]
x_test = x_all[~train_idx]
y_test = y_obs[~train_idx]

x_star = np.linspace(1, 15, 200)
theta = [3, 0.1, 10, 0.2]  # Rational Quadratic kernel parameters

Sigma_prior = calcSigma_RQ(x_star, x_star, theta)
prior_samples = multivariate_normal.rvs(
    mean=np.zeros(len(x_star)),
    cov=Sigma_prior + 1e-6 * np.eye(len(x_star)),
    size=5
)

K_xx = calcSigma_RQ(x_train, x_train, theta)
K_xxs = calcSigma_RQ(x_train, x_star, theta)
K_xsx = K_xxs.T
K_xsxs = calcSigma_RQ(x_star, x_star, theta)

sigma_n = np.sqrt(theta[1])
K_inv = np.linalg.inv(K_xx + sigma_n**2 * np.eye(len(x_train)))

f_star_mean = K_xsx @ K_inv @ y_train
cov_f_star = K_xsxs - K_xsx @ K_inv @ K_xxs

posterior_samples = multivariate_normal.rvs(
    mean=f_star_mean,
    cov=cov_f_star + 1e-6 * np.eye(len(x_star)),
    size=5
)

conf_upper = f_star_mean + 1.96 * np.sqrt(np.diag(cov_f_star))
conf_lower = f_star_mean - 1.96 * np.sqrt(np.diag(cov_f_star))

plt.figure(figsize=(18, 4))

plt.subplot(1, 4, 1)
plt.scatter(x_train, y_train, color='blue', label='Train')
plt.scatter(x_test, y_test, color='red', label='Test')
plt.title('Noisy Observations')
plt.xlabel('x')
plt.ylabel('y')
plt.axvline(12, color='gray', linestyle='--')
plt.legend()
plt.grid(True)

plt.subplot(1, 4, 2)
for i in range(5):
    plt.plot(x_star, prior_samples[i], color='orange')
plt.title('GP PRIOR Samples (RQ)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

plt.subplot(1, 4, 3)
for i in range(5):
    plt.plot(x_star, posterior_samples[i], color='green')
plt.title('GP POSTERIOR Samples (RQ)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

plt.subplot(1, 4, 4)
plt.plot(x_star, f_star_mean, color='blue', label='Posterior Mean')
plt.fill_between(x_star, conf_lower, conf_upper, color='gray', alpha=0.4, label='95% CI')
plt.scatter(x_train, y_train, color='blue', s=10)
plt.scatter(x_test, y_test, color='red', s=10)
plt.title('Posterior Mean + 95% CI (RQ Kernel)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------

#Figure 14
np.random.seed(42)

def calcSigma_RQ(X1, X2, theta):
    v1 = theta[0]  # 信号方差
    v2 = theta[1]  # 噪声方差
    alpha = theta[2]  # shape parameter
    r = theta[3]  # length scale

    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(1, -1)
    sqdist = (X1 - X2) ** 2
    cov = v1 * (1 + sqdist / (2 * alpha * r**2)) ** (-alpha)
    if X1.shape == X2.T.shape:
        cov += v2 * np.eye(len(X1))
    return cov

def true_function(x):
    return np.log(x + 1) + np.cos(3 * np.pi * x)

n_total = 40
x_all = np.sort(np.random.uniform(1, 15, n_total))
y_true = true_function(x_all)
y_obs = y_true + np.random.normal(0, 1, size=n_total)

train_idx = x_all < 12
x_train = x_all[train_idx]
y_train = y_obs[train_idx]
x_test = x_all[~train_idx]
y_test = y_obs[~train_idx]

x_star = np.linspace(1, 15, 200)
theta = [3, 0.1, 10, 0.5]  # Rational Quadratic kernel parameters

Sigma_prior = calcSigma_RQ(x_star, x_star, theta)
prior_samples = multivariate_normal.rvs(
    mean=np.zeros(len(x_star)),
    cov=Sigma_prior + 1e-6 * np.eye(len(x_star)),
    size=5
)

K_xx = calcSigma_RQ(x_train, x_train, theta)
K_xxs = calcSigma_RQ(x_train, x_star, theta)
K_xsx = K_xxs.T
K_xsxs = calcSigma_RQ(x_star, x_star, theta)

sigma_n = np.sqrt(theta[1])
K_inv = np.linalg.inv(K_xx + sigma_n**2 * np.eye(len(x_train)))

f_star_mean = K_xsx @ K_inv @ y_train
cov_f_star = K_xsxs - K_xsx @ K_inv @ K_xxs

posterior_samples = multivariate_normal.rvs(
    mean=f_star_mean,
    cov=cov_f_star + 1e-6 * np.eye(len(x_star)),
    size=5
)

conf_upper = f_star_mean + 1.96 * np.sqrt(np.diag(cov_f_star))
conf_lower = f_star_mean - 1.96 * np.sqrt(np.diag(cov_f_star))

plt.figure(figsize=(18, 4))

plt.subplot(1, 4, 1)
plt.scatter(x_train, y_train, color='blue', label='Train')
plt.scatter(x_test, y_test, color='red', label='Test')
plt.title('Noisy Observations')
plt.xlabel('x')
plt.ylabel('y')
plt.axvline(12, color='gray', linestyle='--')
plt.legend()
plt.grid(True)

plt.subplot(1, 4, 2)
for i in range(5):
    plt.plot(x_star, prior_samples[i], color='orange')
plt.title('GP PRIOR Samples (RQ)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

plt.subplot(1, 4, 3)
for i in range(5):
    plt.plot(x_star, posterior_samples[i], color='green')
plt.title('GP POSTERIOR Samples (RQ)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

plt.subplot(1, 4, 4)
plt.plot(x_star, f_star_mean, color='blue', label='Posterior Mean')
plt.fill_between(x_star, conf_lower, conf_upper, color='gray', alpha=0.4, label='95% CI')
plt.scatter(x_train, y_train, color='blue', s=10)
plt.scatter(x_test, y_test, color='red', s=10)
plt.title('Posterior Mean + 95% CI (RQ Kernel)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------

# Figure15
np.random.seed(42)

def calcSigma_RQ(X1, X2, theta):
    v1 = theta[0]  # 信号方差
    v2 = theta[1]  # 噪声方差
    alpha = theta[2]  # shape parameter
    r = theta[3]  # length scale

    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(1, -1)
    sqdist = (X1 - X2) ** 2
    cov = v1 * (1 + sqdist / (2 * alpha * r**2)) ** (-alpha)
    if X1.shape == X2.T.shape:
        cov += v2 * np.eye(len(X1))
    return cov

def true_function(x):
    return np.log(x + 1) + np.cos(3 * np.pi * x)

n_total = 40
x_all = np.sort(np.random.uniform(1, 15, n_total))
y_true = true_function(x_all)
y_obs = y_true + np.random.normal(0, 1, size=n_total)

train_idx = x_all < 12
x_train = x_all[train_idx]
y_train = y_obs[train_idx]
x_test = x_all[~train_idx]
y_test = y_obs[~train_idx]

x_star = np.linspace(1, 15, 200)
theta = [3, 0.1, 10, 1]  # Rational Quadratic kernel parameters

Sigma_prior = calcSigma_RQ(x_star, x_star, theta)
prior_samples = multivariate_normal.rvs(
    mean=np.zeros(len(x_star)),
    cov=Sigma_prior + 1e-6 * np.eye(len(x_star)),
    size=5
)

K_xx = calcSigma_RQ(x_train, x_train, theta)
K_xxs = calcSigma_RQ(x_train, x_star, theta)
K_xsx = K_xxs.T
K_xsxs = calcSigma_RQ(x_star, x_star, theta)

sigma_n = np.sqrt(theta[1])
K_inv = np.linalg.inv(K_xx + sigma_n**2 * np.eye(len(x_train)))

f_star_mean = K_xsx @ K_inv @ y_train
cov_f_star = K_xsxs - K_xsx @ K_inv @ K_xxs

posterior_samples = multivariate_normal.rvs(
    mean=f_star_mean,
    cov=cov_f_star + 1e-6 * np.eye(len(x_star)),
    size=5
)

conf_upper = f_star_mean + 1.96 * np.sqrt(np.diag(cov_f_star))
conf_lower = f_star_mean - 1.96 * np.sqrt(np.diag(cov_f_star))

plt.figure(figsize=(18, 4))

plt.subplot(1, 4, 1)
plt.scatter(x_train, y_train, color='blue', label='Train')
plt.scatter(x_test, y_test, color='red', label='Test')
plt.title('Noisy Observations')
plt.xlabel('x')
plt.ylabel('y')
plt.axvline(12, color='gray', linestyle='--')
plt.legend()
plt.grid(True)

plt.subplot(1, 4, 2)
for i in range(5):
    plt.plot(x_star, prior_samples[i], color='orange')
plt.title('GP PRIOR Samples (RQ)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

plt.subplot(1, 4, 3)
for i in range(5):
    plt.plot(x_star, posterior_samples[i], color='green')
plt.title('GP POSTERIOR Samples (RQ)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

plt.subplot(1, 4, 4)
plt.plot(x_star, f_star_mean, color='blue', label='Posterior Mean')
plt.fill_between(x_star, conf_lower, conf_upper, color='gray', alpha=0.4, label='95% CI')
plt.scatter(x_train, y_train, color='blue', s=10)
plt.scatter(x_test, y_test, color='red', s=10)
plt.title('Posterior Mean + 95% CI (RQ Kernel)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#-----------------------------------------------------------

#Figure 16
np.random.seed(42)

def calcSigma_RQ(X1, X2, theta):
    v1 = theta[0]  # 信号方差
    v2 = theta[1]  # 噪声方差
    alpha = theta[2]  # shape parameter
    r = theta[3]  # length scale

    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(1, -1)
    sqdist = (X1 - X2) ** 2
    cov = v1 * (1 + sqdist / (2 * alpha * r**2)) ** (-alpha)
    if X1.shape == X2.T.shape:
        cov += v2 * np.eye(len(X1))
    return cov

def true_function(x):
    return np.log(x + 1) + np.cos(3 * np.pi * x)
    
n_total = 40
x_all = np.sort(np.random.uniform(1, 15, n_total))
y_true = true_function(x_all)
y_obs = y_true + np.random.normal(0, 1, size=n_total)

train_idx = x_all < 12
x_train = x_all[train_idx]
y_train = y_obs[train_idx]
x_test = x_all[~train_idx]
y_test = y_obs[~train_idx]

x_star = np.linspace(1, 15, 200)
theta = [3, 0.1, 0.001, 0.234]  # Rational Quadratic kernel parameters

Sigma_prior = calcSigma_RQ(x_star, x_star, theta)
prior_samples = multivariate_normal.rvs(
    mean=np.zeros(len(x_star)),
    cov=Sigma_prior + 1e-6 * np.eye(len(x_star)),
    size=5
)

K_xx = calcSigma_RQ(x_train, x_train, theta)
K_xxs = calcSigma_RQ(x_train, x_star, theta)
K_xsx = K_xxs.T
K_xsxs = calcSigma_RQ(x_star, x_star, theta)

sigma_n = np.sqrt(theta[1])
K_inv = np.linalg.inv(K_xx + sigma_n**2 * np.eye(len(x_train)))

f_star_mean = K_xsx @ K_inv @ y_train
cov_f_star = K_xsxs - K_xsx @ K_inv @ K_xxs

posterior_samples = multivariate_normal.rvs(
    mean=f_star_mean,
    cov=cov_f_star + 1e-6 * np.eye(len(x_star)),
    size=5
)

conf_upper = f_star_mean + 1.96 * np.sqrt(np.diag(cov_f_star))
conf_lower = f_star_mean - 1.96 * np.sqrt(np.diag(cov_f_star))

plt.figure(figsize=(18, 4))

plt.subplot(1, 4, 1)
plt.scatter(x_train, y_train, color='blue', label='Train')
plt.scatter(x_test, y_test, color='red', label='Test')
plt.title('Noisy Observations')
plt.xlabel('x')
plt.ylabel('y')
plt.axvline(12, color='gray', linestyle='--')
plt.legend()
plt.grid(True)

plt.subplot(1, 4, 2)
for i in range(5):
    plt.plot(x_star, prior_samples[i], color='orange')
plt.title('GP PRIOR Samples (RQ)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

plt.subplot(1, 4, 3)
for i in range(5):
    plt.plot(x_star, posterior_samples[i], color='green')
plt.title('GP POSTERIOR Samples (RQ)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

plt.subplot(1, 4, 4)
plt.plot(x_star, f_star_mean, color='blue', label='Posterior Mean')
plt.fill_between(x_star, conf_lower, conf_upper, color='gray', alpha=0.4, label='95% CI')
plt.scatter(x_train, y_train, color='blue', s=10)
plt.scatter(x_test, y_test, color='red', s=10)
plt.title('Posterior Mean + 95% CI (RQ Kernel)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


#--------------------------------------------------------

#Figure 17
np.random.seed(42)

def calcSigma_RQ(X1, X2, theta):
    v1 = theta[0]  
    v2 = theta[1] 
    alpha = theta[2] 
    r = theta[3]  

    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(1, -1)
    sqdist = (X1 - X2) ** 2
    cov = v1 * (1 + sqdist / (2 * alpha * r**2)) ** (-alpha)
    if X1.shape == X2.T.shape:
        cov += v2 * np.eye(len(X1))
    return cov

def true_function(x):
    return np.log(x + 1) + np.cos(3 * np.pi * x)

n_total = 40
x_all = np.sort(np.random.uniform(1, 15, n_total))
y_true = true_function(x_all)
y_obs = y_true + np.random.normal(0, 1, size=n_total)

train_idx = x_all < 12
x_train = x_all[train_idx]
y_train = y_obs[train_idx]
x_test = x_all[~train_idx]
y_test = y_obs[~train_idx]

x_star = np.linspace(1, 15, 200)
theta = [3, 0.1, 0.1, 0.234]  # Rational Quadratic kernel parameters

Sigma_prior = calcSigma_RQ(x_star, x_star, theta)
prior_samples = multivariate_normal.rvs(
    mean=np.zeros(len(x_star)),
    cov=Sigma_prior + 1e-6 * np.eye(len(x_star)),
    size=5
)

K_xx = calcSigma_RQ(x_train, x_train, theta)
K_xxs = calcSigma_RQ(x_train, x_star, theta)
K_xsx = K_xxs.T
K_xsxs = calcSigma_RQ(x_star, x_star, theta)

sigma_n = np.sqrt(theta[1])
K_inv = np.linalg.inv(K_xx + sigma_n**2 * np.eye(len(x_train)))

f_star_mean = K_xsx @ K_inv @ y_train
cov_f_star = K_xsxs - K_xsx @ K_inv @ K_xxs

posterior_samples = multivariate_normal.rvs(
    mean=f_star_mean,
    cov=cov_f_star + 1e-6 * np.eye(len(x_star)),
    size=5
)

conf_upper = f_star_mean + 1.96 * np.sqrt(np.diag(cov_f_star))
conf_lower = f_star_mean - 1.96 * np.sqrt(np.diag(cov_f_star))

plt.figure(figsize=(18, 4))

plt.subplot(1, 4, 1)
plt.scatter(x_train, y_train, color='blue', label='Train')
plt.scatter(x_test, y_test, color='red', label='Test')
plt.title('Noisy Observations')
plt.xlabel('x')
plt.ylabel('y')
plt.axvline(12, color='gray', linestyle='--')
plt.legend()
plt.grid(True)

plt.subplot(1, 4, 2)
for i in range(5):
    plt.plot(x_star, prior_samples[i], color='orange')
plt.title('GP PRIOR Samples (RQ)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

plt.subplot(1, 4, 3)
for i in range(5):
    plt.plot(x_star, posterior_samples[i], color='green')
plt.title('GP POSTERIOR Samples (RQ)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

plt.subplot(1, 4, 4)
plt.plot(x_star, f_star_mean, color='blue', label='Posterior Mean')
plt.fill_between(x_star, conf_lower, conf_upper, color='gray', alpha=0.4, label='95% CI')
plt.scatter(x_train, y_train, color='blue', s=10)
plt.scatter(x_test, y_test, color='red', s=10)
plt.title('Posterior Mean + 95% CI (RQ Kernel)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------

#Figure 18
np.random.seed(42)
def calcSigma_RQ(X1, X2, theta):
    v1 = theta[0]  
    v2 = theta[1]  
    alpha = theta[2] 
    r = theta[3]  

    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(1, -1)
    sqdist = (X1 - X2) ** 2
    cov = v1 * (1 + sqdist / (2 * alpha * r**2)) ** (-alpha)
    if X1.shape == X2.T.shape:
        cov += v2 * np.eye(len(X1))
    return cov

def true_function(x):
    return np.log(x + 1) + np.cos(3 * np.pi * x)

n_total = 40
x_all = np.sort(np.random.uniform(1, 15, n_total))
y_true = true_function(x_all)
y_obs = y_true + np.random.normal(0, 1, size=n_total)

train_idx = x_all < 12
x_train = x_all[train_idx]
y_train = y_obs[train_idx]
x_test = x_all[~train_idx]
y_test = y_obs[~train_idx]

x_star = np.linspace(1, 15, 200)

theta = [3, 0.1, 10000, 0.234]  # Rational Quadratic kernel parameters

Sigma_prior = calcSigma_RQ(x_star, x_star, theta)
prior_samples = multivariate_normal.rvs(
    mean=np.zeros(len(x_star)),
    cov=Sigma_prior + 1e-6 * np.eye(len(x_star)),
    size=5
)

K_xx = calcSigma_RQ(x_train, x_train, theta)
K_xxs = calcSigma_RQ(x_train, x_star, theta)
K_xsx = K_xxs.T
K_xsxs = calcSigma_RQ(x_star, x_star, theta)

sigma_n = np.sqrt(theta[1])
K_inv = np.linalg.inv(K_xx + sigma_n**2 * np.eye(len(x_train)))

f_star_mean = K_xsx @ K_inv @ y_train
cov_f_star = K_xsxs - K_xsx @ K_inv @ K_xxs

posterior_samples = multivariate_normal.rvs(
    mean=f_star_mean,
    cov=cov_f_star + 1e-6 * np.eye(len(x_star)),
    size=5
)

conf_upper = f_star_mean + 1.96 * np.sqrt(np.diag(cov_f_star))
conf_lower = f_star_mean - 1.96 * np.sqrt(np.diag(cov_f_star))

plt.figure(figsize=(18, 4))

plt.subplot(1, 4, 1)
plt.scatter(x_train, y_train, color='blue', label='Train')
plt.scatter(x_test, y_test, color='red', label='Test')
plt.title('Noisy Observations')
plt.xlabel('x')
plt.ylabel('y')
plt.axvline(12, color='gray', linestyle='--')
plt.legend()
plt.grid(True)

plt.subplot(1, 4, 2)
for i in range(5):
    plt.plot(x_star, prior_samples[i], color='orange')
plt.title('GP PRIOR Samples (RQ)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

plt.subplot(1, 4, 3)
for i in range(5):
    plt.plot(x_star, posterior_samples[i], color='green')
plt.title('GP POSTERIOR Samples (RQ)')
plt.xlabel('x')
plt.ylabel('Sampled y')
plt.grid(True)

plt.subplot(1, 4, 4)
plt.plot(x_star, f_star_mean, color='blue', label='Posterior Mean')
plt.fill_between(x_star, conf_lower, conf_upper, color='gray', alpha=0.4, label='95% CI')
plt.scatter(x_train, y_train, color='blue', s=10)
plt.scatter(x_test, y_test, color='red', s=10)
plt.title('Posterior Mean + 95% CI (RQ Kernel)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#-------------------------------------------------------------------

#Figure19
#settings
rng = default_rng(0)          # change to default_rng() for different samples each run
n_samples = 5                 # number of functions sampled in each panel
x = np.linspace(0.0, 5.0, 300)

# Normalize and rescale input to avoid too large values for DotProduct kernel
x_feat = ((x - x.mean()) / 2.0).reshape(-1, 1)

# Base kernels (hyperparameters can be adjusted)
rbf = RBF(length_scale=1.6)
lin = DotProduct(sigma_0=1.0)            # linear kernel (formerly DP)
rq  = RationalQuadratic(length_scale=1.2, alpha=0.8)
per = ExpSineSquared(length_scale=1.0, periodicity=2.2)

# Composite kernels: addition & multiplication
panels = [
    ("RBF+LIN",  rbf + lin),
    ("RBF*LIN",  rbf * lin),
    ("RQ+PER",   rq  + per),
    ("RQ*PER",   rq  * per),
]

def gp_prior_samples(kernel, X, n_funcs, rng):
    """Sample from zero-mean GP prior given kernel and 1D inputs X."""
    K = kernel(X)                      # covariance matrix
    K = K + 1e-8 * np.eye(len(X))      # numerical stability
    Y = rng.multivariate_normal(np.zeros(len(X)), K, size=n_funcs).T
    return Y                           # shape: (len(X), n_funcs)

def style_axes(ax, ylim=(-3.5, 3.5)):
    """Apply consistent axis style for better visualization."""
    for s in ax.spines.values():
        s.set_linewidth(1.4)
    ax.tick_params(axis='both', labelsize=9, width=1.2, length=3)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(*ylim)

#plotting
fig, axes = plt.subplots(2, 2, figsize=(10, 6.2),
                         gridspec_kw={"wspace": 0.35, "hspace": 0.60})

for ax, (label, kernel) in zip(axes.ravel(), panels):
    Y = gp_prior_samples(kernel, x_feat, n_samples, rng)
    for i in range(n_samples):
        ax.plot(x, Y[:, i], linewidth=2.2)
    style_axes(ax)
    # Label y-axis with kernel name (consistent with example figure)
    ax.set_ylabel(label, rotation=90, fontsize=11, labelpad=10)
    ax.set_xlabel("")

plt.savefig("composite_kernels_priors_lin.png", dpi=300, bbox_inches="tight")
plt.show()

#-------------------------------------------------------------------------------------------

#Figure 20
#Hhyperparameters 
sigma2 = 1.0      # amplitude (here all kernels scaled to 1)
ell    = 1.0      # length-scale l
alpha  = 2.0      # RQ shape
p      = 3.0      # PER period

# Distance domain r in [-5, 5]
r = np.linspace(-5.0, 5.0, 800)

# base kernels k(r)
def k_rbf(r, ell=1.0, sigma2=1.0):
    return sigma2 * np.exp(-(r**2)/(2.0*ell**2))

def k_rq(r, ell=1.0, alpha=2.0, sigma2=1.0):
    return sigma2 * (1.0 + (r**2)/(2.0*alpha*ell**2))**(-alpha)

def k_per(r, ell=1.0, p=3.0, sigma2=1.0):
    # ExpSineSquared: sigma^2 * exp( - 2 sin^2(pi r / p) / l^2 )
    return sigma2 * np.exp(-2.0 * np.sin(np.pi * r / p)**2 / (ell**2))

def k_lin(r):
    # Linear kernel section (same as DotProduct without bias)
    return r

# composite kernels on r 
def add(a, b):  return a + b
def mul(a, b):  return a * b

# Compute profiles
rbf   = k_rbf(r, ell=ell, sigma2=sigma2)
rq    = k_rq(r,  ell=ell, alpha=alpha, sigma2=sigma2)
per   = k_per(r, ell=ell, p=p, sigma2=sigma2)
lin   = k_lin(r)

profiles = [
    ("RBF+LIN",  add(rbf, lin)),   ("RBF*LIN",  mul(rbf, lin)),
    ("RBF+PER",  add(rbf, per)),   ("RBF*PER",  mul(rbf, per)),
    ("RQ+LIN",   add(rq,  lin)),   ("RQ*LIN",   mul(rq,  lin)),
]

# plotting 
fig, axes = plt.subplots(3, 2, figsize=(10, 7.2),
                         gridspec_kw={"wspace": 0.35, "hspace": 0.75})

for ax, (label, y) in zip(axes.ravel(), profiles):
    ax.plot(r, y, linewidth=2.3)
    # style
    for s in ax.spines.values():
        s.set_linewidth(1.4)
    ax.tick_params(axis='both', labelsize=9, width=1.2, length=3)
    ax.set_xlim(r.min(), r.max())
    ax.margins(x=0.03)
    ax.set_ylabel(label, rotation=90, fontsize=11, labelpad=10)
    ax.set_xlabel("")

plt.savefig("composite_kernel_profiles_6panels_lin.png", dpi=300, bbox_inches="tight")
plt.show()

#----------------------------------------------------------------------------------

#Figure21
#Hyperparameters
sigma2 = 1.0   # amplitude
ell    = 1.0   # length-scale l
alpha  = 2.0   # RQ shape
p      = 3.0   # PER period

# Domains: symmetric for kernels involving PER; nonnegative for pure distance kernels
r_sym  = np.linspace(-5.0, 5.0, 800)   # for plots with PER
r_pos  = np.linspace(0.0,  5.0, 800)   # for RQ+Matern and RQ*Matern

# Base kernels k(r) 
def k_rbf(r, ell=1.0, sigma2=1.0):
    return sigma2 * np.exp(-(r**2)/(2.0*ell**2))

def k_rq(r, ell=1.0, alpha=2.0, sigma2=1.0):
    # Rational Quadratic as function of distance r >= 0
    return sigma2 * (1.0 + (r**2)/(2.0*alpha*ell**2))**(-alpha)

def k_per(r, ell=1.0, p=3.0, sigma2=1.0):
    # ExpSineSquared: sigma^2 * exp( - 2 sin^2(pi r / p) / l^2 )
    return sigma2 * np.exp(-2.0 * np.sin(np.pi * r / p)**2 / (ell**2))

def k_matern32(r, ell=1.0, sigma2=1.0):
    # Matérn ν=3/2: sigma^2 * (1 + sqrt(3) r / l) * exp(-sqrt(3) r / l)
    c = np.sqrt(3.0) * r / ell
    return sigma2 * (1.0 + c) * np.exp(-c)

def k_lin_section(r):
    # Linear (DotProduct) illustrative section
    return r

# Short-hands using |r| when needed
abs_sym  = np.abs(r_sym)
abs_pos  = r_pos  # already nonnegative

#Composite profiles 
profiles = [
    ("RQ+PER",    k_rq(abs_sym, ell, alpha, sigma2) + k_per(r_sym, ell, p, sigma2),
                  r_sym),
    ("RQ*PER",    k_rq(abs_sym, ell, alpha, sigma2) * k_per(r_sym, ell, p, sigma2),
                  r_sym),
    ("LIN+PER",   k_lin_section(r_sym) + k_per(r_sym, ell, p, sigma2),
                  r_sym),
    ("LIN*PER",   k_lin_section(r_sym) * k_per(r_sym, ell, p, sigma2),
                  r_sym),
    ("RQ+Matern", k_rq(abs_pos, ell, alpha, sigma2) + k_matern32(abs_pos, ell, sigma2),
                  r_pos),
    ("RQ*Matern", k_rq(abs_pos, ell, alpha, sigma2) * k_matern32(abs_pos, ell, sigma2),
                  r_pos),
]

# Plotting 
fig, axes = plt.subplots(3, 2, figsize=(10, 6.6),
                         gridspec_kw={"wspace": 0.35, "hspace": 0.75})

for ax, (label, y, rr) in zip(axes.ravel(), profiles):
    ax.plot(rr, y, linewidth=2.3)
    # styling
    for s in ax.spines.values():
        s.set_linewidth(1.4)
    ax.tick_params(axis='both', labelsize=9, width=1.2, length=3)
    ax.set_xlim(rr.min(), rr.max())
    ax.margins(x=0.03)
    ax.set_ylabel(label, rotation=90, fontsize=11, labelpad=10)
    ax.set_xlabel("")

plt.savefig("composite_kernel_profiles_rq_per_lin_matern.png", dpi=300, bbox_inches="tight")
plt.show()


