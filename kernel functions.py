import numpy as np, matplotlib.pyplot as plt
from numpy.random import default_rng
from sklearn.gaussian_process.kernels import (
    RBF, RationalQuadratic, Matern, ExpSineSquared,
    DotProduct, WhiteKernel, ConstantKernel
)

#----------------------------------------------------------------------------------------------------

#Figure 1
length_scale = 1.0         # RBF核的长度尺度                # Manually adjusted hyperparameters
variance = 1.0             # 核函数前的常数（方差）
n_prior_samples = 5        # 采样条数
n_points = 100             # X轴点数

kernel = ConstantKernel(constant_value=variance) * RBF(length_scale=length_scale)        # Construct kernel function

X = np.linspace(0, 5, n_points).reshape(-1, 1)            # Generate test points (1D input)
K = kernel(X)                                             # Construct covariance matrix
K += 1e-6 * np.eye(n_points)                              # Add small jitter to ensure numerical stability

## Sample GP prior
prior_samples = np.random.multivariate_normal(               
    mean=np.zeros(n_points),
    cov=K,
    size=n_prior_samples
)

#plotting
plt.figure(figsize=(10, 5))
for i in range(n_prior_samples):
    plt.plot(X, prior_samples[i], label=f'Prior Sample {i+1}')
plt.title(f'GP Prior Samples with RBF Kernel\nlength_scale={length_scale}, variance={variance}')
plt.xlabel('Input X')
plt.ylabel('Sampled Function Value')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------

#Figure 2
length_scale = 0.5
alpha = 1000.0
variance = 1.0
n_prior_samples = 5
n_points = 100

kernel = ConstantKernel(variance) * RationalQuadratic(length_scale=length_scale, alpha=alpha)

X = np.linspace(0, 5, n_points).reshape(-1, 1)
K = kernel(X) + 1e-6 * np.eye(n_points)
samples = np.random.multivariate_normal(np.zeros(n_points), K, size=n_prior_samples)

plt.figure(figsize=(10, 5))
for i in range(n_prior_samples):
    plt.plot(X, samples[i], label=f'Sample {i+1}')
plt.title(f'GP Prior Samples with RQ Kernel\nlength_scale={length_scale}, alpha={alpha}, variance={variance}')
plt.xlabel('Input X')
plt.ylabel('Sampled Function Value')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------------

#Figure 3
length_scale = 0.3
nu = 2.5  # 可设为 0.5, 1.5, 2.5
variance = 1.0
n_prior_samples = 5
n_points = 100

kernel = ConstantKernel(variance) * Matern(length_scale=length_scale, nu=nu)

X = np.linspace(0, 5, n_points).reshape(-1, 1)
K = kernel(X) + 1e-6 * np.eye(n_points)
samples = np.random.multivariate_normal(np.zeros(n_points), K, size=n_prior_samples)

plt.figure(figsize=(10, 5))
for i in range(n_prior_samples):
    plt.plot(X, samples[i], label=f'Sample {i+1}')
plt.title(f'GP Prior Samples with Matern Kernel (ν={nu})\nlength_scale={length_scale}, variance={variance}')
plt.xlabel('Input X')
plt.ylabel('Sampled Function Value')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------------

#Figure4
length_scale = 1.0
periodicity = 4.0
variance = 1.0
n_prior_samples = 5
n_points = 100

kernel = ConstantKernel(variance) * ExpSineSquared(length_scale=length_scale, periodicity=periodicity)

X = np.linspace(0, 5, n_points).reshape(-1, 1)
K = kernel(X) + 1e-6 * np.eye(n_points)
samples = np.random.multivariate_normal(np.zeros(n_points), K, size=n_prior_samples)

plt.figure(figsize=(10, 5))
for i in range(n_prior_samples):
    plt.plot(X, samples[i], label=f'Sample {i+1}')
plt.title(f'GP Prior Samples with PER Kernel \nlength_scale={length_scale}, periodicity={periodicity}, variance={variance}')
plt.xlabel('Input X')
plt.ylabel('Sampled Function Value')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------

#Figure 5
sigma_0 = 2.0 
n_prior_samples = 5
n_points = 100

kernel = DotProduct(sigma_0=sigma_0) + WhiteKernel(noise_level=1e-6)

X = np.linspace(0, 5, n_points).reshape(-1, 1)
K = kernel(X)  # DotProduct 自带 bias 项，已经包括常数
samples = np.random.multivariate_normal(np.zeros(n_points), K, size=n_prior_samples)

plt.figure(figsize=(10, 5))
for i in range(n_prior_samples):
    plt.plot(X, samples[i], label=f'Sample {i+1}')
plt.title(f'GP Prior Samples with LIN Kernel\nsigma_0={sigma_0}')
plt.xlabel('Input X')
plt.ylabel('Sampled Function Value')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


