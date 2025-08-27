
#Library Imports 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, RationalQuadratic, Matern, ExpSineSquared,
    DotProduct, WhiteKernel, ConstantKernel
)

#Data Preparation
np.random.seed(40)
df = pd.read_csv(r'E:/系统默认/桌面/eggprice.csv')
data = df['price'].to_numpy()

# Sliding Window and test size
N = 6
test_size = 53  

# Feature Construction
X = np.zeros((len(data) - N, N))
y = np.zeros(len(data) - N)
for i in range(len(data) - N):
    X[i, :] = data[i : i + N]
    y[i]    = data[i + N]

# Train-Test Split
X_train, y_train = X[:-test_size], y[:-test_size]
X_test,  y_test  = X[-test_size:], y[-test_size:]

#------------------------------------------------------------------

#FIgure 23
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) #Kernel Definition

#Model Construction and Training
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=0.01,
    n_restarts_optimizer=10,
    normalize_y=True,
    random_state=40
)
gp.fit(X_train, y_train)

# 95% CI Prediction and Evaluation

y_pred, y_std = gp.predict(X_test, return_std=True)
conf95 = 1.96 * y_std
mse = np.mean((y_pred - y_test) ** 2)
r2  = r2_score(y_test, y_pred)
ape = np.abs((y_test - y_pred) / y_test) * 100  # 转换为百分比
mape = np.mean(ape)

n_prior  = 5   
n_post   = 5   

#Prior Sampling
K_prior = gp.kernel_(X_test)
prior_samples = np.random.multivariate_normal(
    mean=np.zeros(len(X_test)),
    cov=K_prior + 1e-6 * np.eye(len(X_test)),
    size=n_prior
)  # shape (n_prior, n_test)

#Posterior Sampling
posterior_samples = gp.sample_y(
    X_test,
    n_samples=n_post,
    random_state=123
).T  

# plotting
plt.figure(figsize=(18, 4))
plt.subplot(1,4,1)
plt.plot(y_test,  'bo', label='Actual')
plt.xlabel('Test Sample Index')
plt.ylabel('Egg Price')

plt.subplot(1,4,2)
for i in range(n_prior):
    plt.plot(prior_samples[i], lw=1.2, label=f'Prior sample {i+1}')
plt.title(f'Gaussian Process PRIOR  ({n_prior} draws)')
plt.xlabel('Test Sample Index')
plt.ylabel('Sampled Value')
plt.legend(loc='upper right', ncol=3, fontsize='small')
plt.grid(True)
plt.tight_layout()

plt.subplot(1,4,3)

for i in range(n_post):
    plt.plot(posterior_samples[i], lw=1.2, label=f'Posterior sample {i+1}')
plt.title(f'Gaussian Process POSTERIOR  ({n_post} draws)')
plt.xlabel('Test Sample Index')
plt.ylabel('Sampled Value')
plt.legend(loc='upper right', ncol=3, fontsize='small')
plt.grid(True)
plt.tight_layout()

plt.subplot(1,4,4)
plt.plot(y_test,  'bo', label='Actual')
plt.plot(y_pred,  'r-', label='Predicted')
plt.fill_between(
    np.arange(len(y_pred)),
    y_pred - conf95,
    y_pred + conf95,
    color='pink', alpha=0.3, label='95% CI'
)
plt.title(f"Kernel: {gp.kernel_}")
plt.xlabel('Test Sample Index')
plt.ylabel('Egg Price')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(6, 4))
plt.plot(range(len(ape)), ((y_test - y_pred) / y_test), color='orange', alpha=0.7)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('  Error (%)', fontsize=12)
plt.title('Prediction Error ', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout() 
plt.show()

# Print Results 
print(f"\nRMSE = {mse:.2f}")
print(f"\nMAPE = {mape:.2f}")
print(f"\nR2 = {r2:.2f}")
print("\nOptimized kernel:", gp.kernel_)

#----------------------------------------------------------------------------

# Figure 24
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=0.01,
    n_restarts_optimizer=10,
    normalize_y=True,
    random_state=40
)
gp.fit(X_train, y_train)

y_pred, y_std = gp.predict(X_test, return_std=True)
conf95 = 1.96 * y_std
mse = np.mean((y_pred - y_test) ** 2)
r2  = r2_score(y_test, y_pred)
ape = np.abs((y_test - y_pred) / y_test) * 100  

n_prior  = 5   
n_post   = 5   

K_prior = gp.kernel_(X_test)
prior_samples = np.random.multivariate_normal(
    mean=np.zeros(len(X_test)),
    cov=K_prior + 1e-6 * np.eye(len(X_test)),
    size=n_prior
)  # shape (n_prior, n_test)

posterior_samples = gp.sample_y(
    X_test,
    n_samples=n_post,
    random_state=123
).T  

plt.figure(figsize=(18, 4))
plt.subplot(1,4,1)
plt.plot(y_test,  'bo', label='Actual')
plt.xlabel('Test Sample Index')
plt.ylabel('Egg Price')

plt.subplot(1,4,2)
for i in range(n_prior):
    plt.plot(prior_samples[i], lw=1.2, label=f'Prior sample {i+1}')
plt.title(f'Gaussian Process PRIOR  ({n_prior} draws)')
plt.xlabel('Test Sample Index')
plt.ylabel('Sampled Value')
plt.legend(loc='upper right', ncol=3, fontsize='small')
plt.grid(True)
plt.tight_layout()

plt.subplot(1,4,3)
for i in range(n_post):
    plt.plot(posterior_samples[i], lw=1.2, label=f'Posterior sample {i+1}')
plt.title(f'Gaussian Process POSTERIOR  ({n_post} draws)')
plt.xlabel('Test Sample Index')
plt.ylabel('Sampled Value')
plt.legend(loc='upper right', ncol=3, fontsize='small')
plt.grid(True)
plt.tight_layout()

plt.subplot(1,4,4)
plt.plot(y_test,  'bo', label='Actual')
plt.plot(y_pred,  'r-', label='Predicted')
plt.fill_between(
    np.arange(len(y_pred)),
    y_pred - conf95,
    y_pred + conf95,
    color='pink', alpha=0.3, label='95% CI'
)
plt.title(f"Kernel: {gp.kernel_}")
plt.xlabel('Test Sample Index')
plt.ylabel('Egg Price')
plt.legend()
plt.grid(True)
plt.tight_layout()


plt.figure(figsize=(6, 4))
plt.plot(range(len(ape)), ((y_test - y_pred) / y_test), color='orange', alpha=0.7)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('  Error (%)', fontsize=12)
plt.title('Prediction Error ', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()  # 自动调整子图间距
plt.show()

print(f"\nRMSE = {mse:.2f}")
print(f"\nMAPE = {mape:.2f}")
print(f"\nR2 = {r2:.2f}")
print("\nOptimized kernel:", gp.kernel_)

#---------------------------------------------------------------------------------------------------------

# Figure 25
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=0.01,
    n_restarts_optimizer=10,
    normalize_y=True,
    random_state=40
)
gp.fit(X_train, y_train)

y_pred, y_std = gp.predict(X_test, return_std=True)
conf95 = 1.96 * y_std
mse = np.mean((y_pred - y_test) ** 2)
r2  = r2_score(y_test, y_pred)
ape = np.abs((y_test - y_pred) / y_test) * 100  # 转换为百分比
mape = np.mean(ape)

n_prior  = 5   
n_post   = 5  

K_prior = gp.kernel_(X_test)
prior_samples = np.random.multivariate_normal(
    mean=np.zeros(len(X_test)),
    cov=K_prior + 1e-6 * np.eye(len(X_test)),
    size=n_prior
)  # shape (n_prior, n_test)

posterior_samples = gp.sample_y(
    X_test,
    n_samples=n_post,
    random_state=123
).T  

plt.figure(figsize=(18, 4))
plt.subplot(1,4,1)
plt.plot(y_test,  'bo', label='Actual')
plt.xlabel('Test Sample Index')
plt.ylabel('Egg Price')

plt.subplot(1,4,2)
for i in range(n_prior):
    plt.plot(prior_samples[i], lw=1.2, label=f'Prior sample {i+1}')
plt.title(f'Gaussian Process PRIOR  ({n_prior} draws)')
plt.xlabel('Test Sample Index')
plt.ylabel('Sampled Value')
plt.legend(loc='upper right', ncol=3, fontsize='small')
plt.grid(True)
plt.tight_layout()

plt.subplot(1,4,3)
for i in range(n_post):
    plt.plot(posterior_samples[i], lw=1.2, label=f'Posterior sample {i+1}')
plt.title(f'Gaussian Process POSTERIOR  ({n_post} draws)')
plt.xlabel('Test Sample Index')
plt.ylabel('Sampled Value')
plt.legend(loc='upper right', ncol=3, fontsize='small')
plt.grid(True)
plt.tight_layout()

plt.subplot(1,4,4)
plt.plot(y_test,  'bo', label='Actual')
plt.plot(y_pred,  'r-', label='Predicted')
plt.fill_between(
    np.arange(len(y_pred)),
    y_pred - conf95,
    y_pred + conf95,
    color='pink', alpha=0.3, label='95% CI'
)
plt.title(f"Kernel: {gp.kernel_}")
plt.xlabel('Test Sample Index')
plt.ylabel('Egg Price')
plt.legend()
plt.grid(True)
plt.tight_layout()


plt.figure(figsize=(6, 4))
plt.plot(range(len(ape)), ((y_test - y_pred) / y_test), color='orange', alpha=0.7)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('  Error (%)', fontsize=12)
plt.title('Prediction Error ', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()  # 自动调整子图间距
plt.show()

print(f"\nRMSE = {mse:.2f}")
print(f"\nMAPE = {mape:.2f}")
print(f"\nR2 = {r2:.2f}")
print("\nOptimized kernel:", gp.kernel_)

# ---------------------------------------------------------------------------

# Figure 26
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * (
    RBF(length_scale=1.0) +
    RationalQuadratic(length_scale=1.0, alpha=1.0) * DotProduct()
)

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,
    n_restarts_optimizer=10,
    normalize_y=True,
    random_state=40
)
gp.fit(X_train, y_train)

y_pred, y_std = gp.predict(X_test, return_std=True)
conf95 = 1.96 * y_std
mse = np.mean((y_pred - y_test) ** 2)
r2 = r2_score(y_test, y_pred)
ape = np.abs((y_test - y_pred) / y_test) * 100
mape = np.mean(ape)

n_prior = 5
n_post = 5

K_prior = gp.kernel_(X_test)
prior_samples = np.random.multivariate_normal(
    mean=np.zeros(len(X_test)),
    cov=K_prior + 1e-6 * np.eye(len(X_test)),
    size=n_prior
)

posterior_samples = gp.sample_y(X_test, n_samples=n_post, random_state=123).T

plt.figure(figsize=(18, 4))
plt.subplot(1, 4, 1)
plt.plot(y_test, 'bo', label='Actual')
plt.xlabel('Test Sample Index')
plt.ylabel('Egg Price')

plt.subplot(1, 4, 2)
for i in range(n_prior):
    plt.plot(prior_samples[i], lw=1.2, label=f'Prior sample {i+1}')
plt.title(f'GP PRIOR Samples ({n_prior} draws)')
plt.xlabel('Test Sample Index')
plt.ylabel('Sampled Value')
plt.legend(loc='upper right', ncol=3, fontsize='small')
plt.grid(True)
plt.tight_layout()

plt.subplot(1, 4, 3)
for i in range(n_post):
    plt.plot(posterior_samples[i], lw=1.2, label=f'Posterior sample {i+1}')
plt.title(f'GP POSTERIOR Samples ({n_post} draws)')
plt.xlabel('Test Sample Index')
plt.ylabel('Sampled Value')
plt.legend(loc='upper right', ncol=3, fontsize='small')
plt.grid(True)
plt.tight_layout()

plt.subplot(1, 4, 4)
plt.plot(y_test, 'bo', label='Actual')
plt.plot(y_pred, 'r-', label='Predicted')
plt.fill_between(
    np.arange(len(y_pred)),
    y_pred - conf95,
    y_pred + conf95,
    color='pink', alpha=0.3, label='95% CI'
)
plt.title(f'GP‑RBF + (RQ×LIN) | MSE={mse:.2f}, R²={r2:.2f}')
plt.xlabel('Test Sample Index')
plt.ylabel('Egg Price')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(6, 4))
plt.plot(range(len(ape)), ((y_test - y_pred) / y_test), color='orange', alpha=0.7)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Error (%)', fontsize=12)
plt.title('Prediction Error', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print(f"\nRMSE = {mse:.2f}")
print(f"\nMAPE = {mape:.2f}")
print(f"\nR2 = {r2:.2f}")
print("\nOptimized kernel:", gp.kernel_)


#---------------------------------------------------------------------------------------

#Figure 27
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct, ConstantKernel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Data preparation
df = pd.read_csv('E:/系统默认/桌面/eggprice.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
price_series = df['price']
data = price_series.to_numpy()

#SARIMA model
train_ts = price_series[:-53]
test_ts = price_series[-53:]

sarima_model = SARIMAX(train_ts, order=(1, 1, 1), seasonal_order=(1, 0, 1, 52))
sarima_result = sarima_model.fit(disp=False)
sarima_forecast = sarima_result.get_forecast(steps=53)
sarima_pred = sarima_forecast.predicted_mean
sarima_ci = sarima_forecast.conf_int(alpha=0.05)

#  GPR model
N = 6
X = np.zeros((len(data) - N, N))
y = np.zeros(len(data) - N)
for i in range(len(data) - N):
    X[i, :] = data[i : i + N]
    y[i]    = data[i + N]

X_train, y_train = X[:-53], y[:-53]
X_test,  y_test  = X[-53:], y[-53:]

kernel = ConstantKernel(1.0, (1e-3, 1e3)) * (
    RBF(length_scale=1.0) +
    RationalQuadratic(length_scale=1.0, alpha=1.0) * DotProduct()
)

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,
    n_restarts_optimizer=10,
    normalize_y=True,
    random_state=40
)
gp.fit(X_train, y_train)
y_pred_gpr, y_std_gpr = gp.predict(X_test, return_std=True)
conf95_gpr = 1.96 * y_std_gpr

# SARIMA
sarima_rmse = mean_squared_error(test_ts, sarima_pred, squared=False)
sarima_mae = mean_absolute_error(test_ts, sarima_pred)
sarima_r2 = r2_score(test_ts, sarima_pred)
sarima_mape = np.mean(np.abs((test_ts - sarima_pred) / test_ts)) * 100

# GPR
gpr_rmse = mean_squared_error(y_test, y_pred_gpr, squared=False)
gpr_mae = mean_absolute_error(y_test, y_pred_gpr)
gpr_r2 = r2_score(y_test, y_pred_gpr)
gpr_mape = np.mean(np.abs((y_test - y_pred_gpr) / y_test)) * 100

# plotting
plt.figure(figsize=(12, 5))
plt.plot(range(len(y_test)), y_test, 'ko', label='Actual Price')

# SARIMA
plt.plot(range(len(sarima_pred)), sarima_pred.values, 'g-', label='SARIMA Prediction')
plt.fill_between(range(len(sarima_pred)),
                 sarima_ci.iloc[:, 0],
                 sarima_ci.iloc[:, 1],
                 color='green', alpha=0.2, label='SARIMA 95% CI')

# GPR
plt.plot(range(len(y_pred_gpr)), y_pred_gpr, 'r-', label='GPR Prediction')
plt.fill_between(range(len(y_pred_gpr)),
                 y_pred_gpr - conf95_gpr,
                 y_pred_gpr + conf95_gpr,
                 color='red', alpha=0.2, label='GPR 95% CI')

plt.title('Comparison of GPR vs SARIMA Predictions')
plt.xlabel('Test Sample Index')
plt.ylabel('Egg Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print results
print("📊 GPR Performance:")
print(f"  RMSE = {gpr_rmse:.2f}")
print(f"  MAE  = {gpr_mae:.2f}")
print(f"  R²   = {gpr_r2:.2f}")
print(f"  MAPE = {gpr_mape:.2f}%")

print("\n📊 SARIMA Performance:")
print(f"  RMSE = {sarima_rmse:.2f}")
print(f"  MAE  = {sarima_mae:.2f}")
print(f"  R²   = {sarima_r2:.2f}")
print(f"  MAPE = {sarima_mape:.2f}%")


