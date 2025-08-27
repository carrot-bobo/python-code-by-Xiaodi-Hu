#!/usr/bin/env python
# coding: utf-8

# ##RBF

# In[13]:


#Library Imports 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


# In[14]:


#数据准备  Data Preparation
np.random.seed(40)
df = pd.read_csv(r'E:/系统默认/桌面/eggprice.csv')
data = df['price'].to_numpy()

# 设置滑动窗口长度和测试集大小 Sliding Window and test size
N = 6
test_size = 53  # 使用最后 53 个数据点作为测试集

# 构造滑动窗口特征和目标变量 Feature Construction
X = np.zeros((len(data) - N, N))
y = np.zeros(len(data) - N)
for i in range(len(data) - N):
    X[i, :] = data[i : i + N]
    y[i]    = data[i + N]

# 划分训练集和测试集 Train-Test Split
X_train, y_train = X[:-test_size], y[:-test_size]
X_test,  y_test  = X[-test_size:], y[-test_size:]


# In[15]:


# 建立并训练 Gaussian Process 模型

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

# 95% 置信区间预测 Prediction and Evaluation

y_pred, y_std = gp.predict(X_test, return_std=True)
conf95 = 1.96 * y_std
mse = np.mean((y_pred - y_test) ** 2)
r2  = r2_score(y_test, y_pred)
# 计算绝对百分比误差
ape = np.abs((y_test - y_pred) / y_test) * 100  # 转换为百分比
# 计算MAPE
mape = np.mean(ape)

# 先验、后验采样 Prior and Posterior Samples

n_prior  = 5   # 先验曲线数量
n_post   = 5   # 后验曲线数量

#先验采样
K_prior = gp.kernel_(X_test)
prior_samples = np.random.multivariate_normal(
    mean=np.zeros(len(X_test)),
    cov=K_prior + 1e-6 * np.eye(len(X_test)),
    size=n_prior
)  # shape (n_prior, n_test)

#后验采样
posterior_samples = gp.sample_y(
    X_test,
    n_samples=n_post,
    random_state=123
).T  


# In[16]:


#绘图 plotting
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
#后验样本曲线
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


# In[17]:


# 打印结果 Result Output
print(f"\nRMSE = {mse:.2f}")
print(f"\nMAPE = {mape:.2f}")
print(f"\nR2 = {r2:.2f}")

# 打印优化后的核函数参数
print("\nOptimized kernel:", gp.kernel_)


# In[ ]:





# In[18]:


# Library Imports 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

# 数据准备 Data Preparation
np.random.seed(40)
df = pd.read_csv(r'E:/系统默认/桌面/eggprice.csv')  # 请替换为你的实际文件路径
data = df['price'].to_numpy()

# 设置滑动窗口长度和测试集大小
N = 6
test_size = 53

# 构造滑动窗口特征和目标变量 Feature Construction
X = np.zeros((len(data) - N, N))
y = np.zeros(len(data) - N)
for i in range(len(data) - N):
    X[i, :] = data[i : i + N]
    y[i]    = data[i + N]

# 划分训练集和测试集 Train-Test Split
X_train, y_train = X[:-test_size], y[:-test_size]
X_test,  y_test  = X[-test_size:], y[-test_size:]

# 主核函数（模型使用）带自动优化
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))          + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

# 构建高斯过程模型（带优化）
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=0.0,  # 用 WhiteKernel 代替 alpha
    normalize_y=True,
    n_restarts_optimizer=10,
    random_state=40
)

# 模型训练
gp.fit(X_train, y_train)

# 模型预测
y_pred, y_std = gp.predict(X_test, return_std=True)
conf95 = 1.96 * y_std
mse = np.mean((y_pred - y_test) ** 2)
r2  = r2_score(y_test, y_pred)
ape = np.abs((y_test - y_pred) / y_test) * 100
mape = np.mean(ape)

# ===== ✅ 先验采样用手动核函数 =====
manual_prior_kernel = ConstantKernel(1.0) * RBF(length_scale=5.0)  # 可尝试 2~10
K_prior = manual_prior_kernel(X_test)
prior_samples = np.random.multivariate_normal(
    mean=np.zeros(len(X_test)),
    cov=K_prior + 1e-6 * np.eye(len(X_test)),
    size=5
)

# 后验采样（仍使用模型）
posterior_samples = gp.sample_y(X_test, n_samples=5, random_state=123).T

# 绘图
plt.figure(figsize=(18, 4))

# 图1：真实值
plt.subplot(1, 4, 1)
plt.plot(y_test, 'bo', label='Actual')
plt.xlabel('Test Sample Index')
plt.ylabel('Egg Price')

# 图2：先验样本（手动核）
plt.subplot(1, 4, 2)
for i in range(5):
    plt.plot(prior_samples[i], lw=1.2, label=f'Prior sample {i+1}')
plt.title(f'Gaussian Process PRIOR (manual kernel)')
plt.xlabel('Test Sample Index')
plt.ylabel('Sampled Value')
plt.legend(loc='upper right', ncol=3, fontsize='small')
plt.grid(True)

# 图3：后验样本（模型核）
plt.subplot(1, 4, 3)
for i in range(5):
    plt.plot(posterior_samples[i], lw=1.2, label=f'Posterior sample {i+1}')
plt.title(f'Gaussian Process POSTERIOR (model kernel)')
plt.xlabel('Test Sample Index')
plt.ylabel('Sampled Value')
plt.legend(loc='upper right', ncol=3, fontsize='small')
plt.grid(True)

# 图4：预测 + 置信区间
plt.subplot(1, 4, 4)
plt.plot(y_test, 'bo', label='Actual')
plt.plot(y_pred, 'r-', label='Predicted')
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

# 图5：误差图
plt.figure(figsize=(6, 4))
plt.plot(range(len(ape)), ((y_test - y_pred) / y_test), color='orange', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Error (%)')
plt.title('Prediction Error')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 打印评估指标
print(f"\nRMSE = {mse:.2f}")
print(f"MAPE = {mape:.2f}")
print(f"R²    = {r2:.2f}")

# 输出优化后核结构
print("\nOptimized kernel:", gp.kernel_)


# In[ ]:





# ##Matern

# In[101]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel


# In[102]:


#数据准备
np.random.seed(40)
df = pd.read_csv(r'E:/系统默认/桌面/eggprice.csv')
data = df['price'].to_numpy()

# 设置滑动窗口长度和测试集大小
N = 6
test_size = 53  # 使用最后 53 个数据点作为测试集

# 构造滑动窗口特征和目标变量
X = np.zeros((len(data) - N, N))
y = np.zeros(len(data) - N)
for i in range(len(data) - N):
    X[i, :] = data[i : i + N]
    y[i]    = data[i + N]

# 划分训练集和测试集
X_train, y_train = X[:-test_size], y[:-test_size]
X_test,  y_test  = X[-test_size:], y[-test_size:]


# In[103]:


# 建立并训练 Gaussian Process 模型

kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=0.01,
    n_restarts_optimizer=10,
    normalize_y=True,
    random_state=40
)
gp.fit(X_train, y_train)

# 95% 置信区间预测

y_pred, y_std = gp.predict(X_test, return_std=True)
conf95 = 1.96 * y_std
mse = np.mean((y_pred - y_test) ** 2)
r2  = r2_score(y_test, y_pred)
# 计算绝对百分比误差
ape = np.abs((y_test - y_pred) / y_test) * 100  # 转换为百分比
# 计算MAPE
mape = np.mean(ape)

# 先验、后验采样

n_prior  = 5   # 先验曲线数量
n_post   = 5   # 后验曲线数量

#先验采样
K_prior = gp.kernel_(X_test)
prior_samples = np.random.multivariate_normal(
    mean=np.zeros(len(X_test)),
    cov=K_prior + 1e-6 * np.eye(len(X_test)),
    size=n_prior
)  # shape (n_prior, n_test)

#后验采样
posterior_samples = gp.sample_y(
    X_test,
    n_samples=n_post,
    random_state=123
).T  


# In[104]:


#绘图
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
#后验样本曲线
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


# In[105]:


# 打印结果
print(f"\nRMSE = {mse:.2f}")
print(f"\nMAPE = {mape:.2f}")
print(f"\nR2 = {r2:.2f}")

# 打印优化后的核函数参数
print("\nOptimized kernel:", gp.kernel_)


# In[ ]:





# ##RQ

# In[106]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel


# In[107]:


#数据准备
np.random.seed(40)
df = pd.read_csv(r'E:/系统默认/桌面/eggprice.csv')
data = df['price'].to_numpy()

# 设置滑动窗口长度和测试集大小
N = 6
test_size = 53  # 使用最后 53 个数据点作为测试集

# 构造滑动窗口特征和目标变量
X = np.zeros((len(data) - N, N))
y = np.zeros(len(data) - N)
for i in range(len(data) - N):
    X[i, :] = data[i : i + N]
    y[i]    = data[i + N]

# 划分训练集和测试集
X_train, y_train = X[:-test_size], y[:-test_size]
X_test,  y_test  = X[-test_size:], y[-test_size:]


# 建立并训练 Gaussian Process 模型

kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=0.01,
    n_restarts_optimizer=10,
    normalize_y=True,
    random_state=40
)
gp.fit(X_train, y_train)


# 95% 置信区间预测

y_pred, y_std = gp.predict(X_test, return_std=True)
conf95 = 1.96 * y_std
mse = np.mean((y_pred - y_test) ** 2)
r2  = r2_score(y_test, y_pred)
# 计算绝对百分比误差
ape = np.abs((y_test - y_pred) / y_test) * 100  # 转换为百分比
# 计算MAPE
mape = np.mean(ape)

# 先验、后验采样

n_prior  = 5   # 先验曲线数量
n_post   = 5   # 后验曲线数量

#先验采样
K_prior = gp.kernel_(X_test)
prior_samples = np.random.multivariate_normal(
    mean=np.zeros(len(X_test)),
    cov=K_prior + 1e-6 * np.eye(len(X_test)),
    size=n_prior
)  # shape (n_prior, n_test)

#后验采样
posterior_samples = gp.sample_y(
    X_test,
    n_samples=n_post,
    random_state=123
).T  


# In[108]:


#绘图
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
#后验样本曲线
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


# In[109]:


# 打印结果
print(f"\nRMSE = {mse:.2f}")
print(f"\nMAPE = {mape:.2f}")
print(f"\nR2 = {r2:.2f}")

# 打印优化后的核函数参数
print("\nOptimized kernel:", gp.kernel_)


# In[ ]:





# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, ConstantKernel

# 数据准备
np.random.seed(40)
df = pd.read_csv(r'E:/系统默认/桌面/eggprice.csv')  # 替换为你的路径
data = df['price'].to_numpy()

# 设置滑动窗口长度和测试集大小
N = 6
test_size = 53

# 构造滑动窗口特征和目标变量
X = np.zeros((len(data) - N, N))
y = np.zeros(len(data) - N)
for i in range(len(data) - N):
    X[i, :] = data[i : i + N]
    y[i]    = data[i + N]

# 划分训练集和测试集
X_train, y_train = X[:-test_size], y[:-test_size]
X_test,  y_test  = X[-test_size:], y[-test_size:]

# 创建改进后的周期核函数（更合理的初始值 + 边界）
kernel = ConstantKernel(1.0, (1e-2, 1e2)) * ExpSineSquared(
    length_scale=2.0, periodicity=26.0,  # 半年为周期
    length_scale_bounds=(1e-1, 1e2),
    periodicity_bounds=(5.0, 100.0)
)

# 构建 GPR 模型并拟合
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-2,  # 增大 alpha 提升数值稳定性
    n_restarts_optimizer=10,
    normalize_y=True,
    random_state=40
)
gp.fit(X_train, y_train)

# 预测及置信区间
y_pred, y_std = gp.predict(X_test, return_std=True)
conf95 = 1.96 * y_std
mse = np.mean((y_pred - y_test) ** 2)
r2  = r2_score(y_test, y_pred)
ape = np.abs((y_test - y_pred) / y_test) * 100
mape = np.mean(ape)

# 先验采样
n_prior = 5
K_prior = gp.kernel_(X_test)
prior_samples = np.random.multivariate_normal(
    mean=np.zeros(len(X_test)),
    cov=K_prior + 1e-6 * np.eye(len(X_test)),
    size=n_prior
)

# 后验采样
n_post = 5
posterior_samples = gp.sample_y(X_test, n_samples=n_post, random_state=123).T

# 图1：真实 vs 预测
plt.figure(figsize=(18, 4))
plt.subplot(1, 4, 1)
plt.plot(y_test,  'bo', label='Actual')
plt.xlabel('Test Sample Index')
plt.ylabel('Pig Price')

# 图2：先验样本
plt.subplot(1, 4, 2)
for i in range(n_prior):
    plt.plot(prior_samples[i], lw=1.2, label=f'Prior sample {i+1}')
plt.title(f'GP PRIOR  ({n_prior} draws)')
plt.xlabel('Test Sample Index')
plt.ylabel('Sampled Value')
plt.legend(loc='upper right', ncol=3, fontsize='small')
plt.grid(True)

# 图3：后验样本
plt.subplot(1, 4, 3)
for i in range(n_post):
    plt.plot(posterior_samples[i], lw=1.2, label=f'Posterior sample {i+1}')
plt.title(f'GP POSTERIOR  ({n_post} draws)')
plt.xlabel('Test Sample Index')
plt.ylabel('Sampled Value')
plt.legend(loc='upper right', ncol=3, fontsize='small')
plt.grid(True)

# 图4：预测+置信区间
plt.subplot(1, 4, 4)
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
plt.ylabel('Pig Price')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 图5：误差图
plt.figure(figsize=(6, 4))
plt.plot(range(len(ape)), ((y_test - y_pred) / y_test), color='orange', alpha=0.7)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Error (%)', fontsize=12)
plt.title('Prediction Error', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 打印评估指标
print(f"\nRMSE = {mse:.2f}")
print(f"MAPE = {mape:.2f}")
print(f"R2 = {r2:.2f}")

# 打印优化后的核函数结构
print("\nOptimized kernel:", gp.kernel_)


# In[ ]:





# In[ ]:





# ##RBF + (RQ × LIN)

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct, ConstantKernel

#数据准备
np.random.seed(40)
df = pd.read_csv(r'E:/系统默认/桌面/eggprice.csv')
data = df['price'].to_numpy()

# 设置滑动窗口长度和测试集大小
N = 6
test_size = 53  # 使用最后 53 个数据点作为测试集

# 构造滑动窗口特征和目标变量
X = np.zeros((len(data) - N, N))
y = np.zeros(len(data) - N)
for i in range(len(data) - N):
    X[i, :] = data[i : i + N]
    y[i]    = data[i + N]

# 划分训练集和测试集
X_train, y_train = X[:-test_size], y[:-test_size]
X_test,  y_test  = X[-test_size:], y[-test_size:]
# 核函数设定：RBF + (RQ × LIN)
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * (
    RBF(length_scale=1.0) +
    RationalQuadratic(length_scale=1.0, alpha=1.0) * DotProduct()
)

# 构建并训练模型
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,
    n_restarts_optimizer=10,
    normalize_y=True,
    random_state=40
)
gp.fit(X_train, y_train)

# 预测与评估
y_pred, y_std = gp.predict(X_test, return_std=True)
conf95 = 1.96 * y_std
mse = np.mean((y_pred - y_test) ** 2)
r2 = r2_score(y_test, y_pred)
ape = np.abs((y_test - y_pred) / y_test) * 100
mape = np.mean(ape)

# 先验与后验采样
n_prior = 5
n_post = 5

K_prior = gp.kernel_(X_test)
prior_samples = np.random.multivariate_normal(
    mean=np.zeros(len(X_test)),
    cov=K_prior + 1e-6 * np.eye(len(X_test)),
    size=n_prior
)

posterior_samples = gp.sample_y(X_test, n_samples=n_post, random_state=123).T

# 可视化
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

# 打印结果
print(f"\nRMSE = {mse:.2f}")
print(f"\nMAPE = {mape:.2f}")
print(f"\nR2 = {r2:.2f}")

# 打印优化后的核函数参数
print("\nOptimized kernel:", gp.kernel_)


# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 读取数据
df = pd.read_csv('E:/系统默认/桌面/eggprice.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
ts = df['price']

# 划分训练集和测试集（后53个为测试集）
train = ts[:-53]
test = ts[-53:]

# 拟合 SARIMA 模型
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 52))
result = model.fit(disp=False)

# 预测及置信区间（95%）
forecast_result = result.get_forecast(steps=53)
forecast_mean = forecast_result.predicted_mean
conf_int = forecast_result.conf_int(alpha=0.05)

# 可视化
plt.figure(figsize=(12, 5))
plt.plot(train, label='Training Set')
plt.plot(test, label='Actual Test', color='orange')
plt.plot(forecast_mean, label='Forecast', color='green')
plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                 color='green', alpha=0.3, label='95% Confidence Interval')
plt.title('Egg Price Forecast with SARIMA and 95% CI')
plt.xlabel('Date')
plt.ylabel('Egg Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 评估指标（保留两位小数）
rmse = mean_squared_error(test, forecast_mean, squared=False)
mae = mean_absolute_error(test, forecast_mean)
r2 = r2_score(test, forecast_mean)

print(f"RMSE = {rmse:.2f}")
print(f"MAE  = {mae:.2f}")
print(f"R²   = {r2:.2f}")


# In[ ]:





# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct, ConstantKernel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========== 数据准备 ==========
df = pd.read_csv('E:/系统默认/桌面/eggprice.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
price_series = df['price']
data = price_series.to_numpy()

# ========== SARIMA 模型 ==========
train_ts = price_series[:-53]
test_ts = price_series[-53:]

sarima_model = SARIMAX(train_ts, order=(1, 1, 1), seasonal_order=(1, 0, 1, 52))
sarima_result = sarima_model.fit(disp=False)
sarima_forecast = sarima_result.get_forecast(steps=53)
sarima_pred = sarima_forecast.predicted_mean
sarima_ci = sarima_forecast.conf_int(alpha=0.05)

# ========== GPR 模型 ==========
N = 6
X = np.zeros((len(data) - N, N))
y = np.zeros(len(data) - N)
for i in range(len(data) - N):
    X[i, :] = data[i : i + N]
    y[i]    = data[i + N]

# 滑动窗口划分训练和测试
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

# ========== 评估指标 ==========
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

# ========== 可视化 ==========
plt.figure(figsize=(12, 5))
plt.plot(range(len(y_test)), y_test, 'ko', label='Actual Price')

# SARIMA（注意：这里是连续时间预测）
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

# ========== 输出评估指标 ==========
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


# In[ ]:





# In[ ]:





# In[ ]:




