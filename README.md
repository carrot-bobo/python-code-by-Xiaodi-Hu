# Gaussian Process Regression and Classification for Egg Price Forecasting

This repository contains Python implementations of **Gaussian Process Regression (GPR)** and **Gaussian Process Classification (GPC)** applied to agricultural commodity price forecasting, with a focus on egg prices in China.  
The project accompanies an MSc dissertation on contemporary regression methods.

---

## Repository Structure

- **kernel functions.py**  
  Demonstrates prior samples from Gaussian processes with different kernels (RBF, RQ, Matern, Periodic, Linear).

- **The Influence of the Kernel Function on the Predictive Distribution.py**  
  Illustrates how different kernels influence GP predictive distributions, including prior and posterior sampling and analytic kernel profiles.

- **Results of GPR.py**  
  Applies GPR to weekly egg price data.  
  - Implements multiple kernels (RBF, RQ, Matern, composite kernels).  
  - Produces prior/posterior samples, predictive means with 95% CI.  
  - Evaluates performance with RMSE, MAPE, and R².

- **Gaussian Classification.py**  
  Extends Gaussian processes to classification tasks.  
  - Synthetic example with nonlinear latent function.  
  - Real data experiments on egg prices for predicting directional movements (up/down).  
  - Kernels tested: RBF, RQ, Matérn, Linear, and composites.  
  - Evaluation metrics: Accuracy, ROC-AUC, Brier score, Confusion Matrix.

---

## Code–Figure Mapping

### 1) `kernel functions.py` — GP priors for single kernels

- **Figure 1 (GP Prior, RBF)**
```python
kernel = ConstantKernel(variance) * RBF(length_scale=...)
K = kernel(X) + 1e-6 * np.eye(len(X))
prior_samples = np.random.multivariate_normal(
    mean=np.zeros(n_points), cov=K, size=n_prior_samples
)
```

- **Figure 2 (RQ)** – same as Fig. 1 with `RationalQuadratic`  
- **Figure 3 (Matern, ν=2.5)** – same as Fig. 1 with `Matern`  
- **Figure 4 (Periodic)** – same as Fig. 1 with `ExpSineSquared`  
- **Figure 5 (Linear)** – same as Fig. 1 with `DotProduct + WhiteKernel`  

---

### 2) `The Influence of the Kernel Function on the Predictive Distribution.py`

#### 2.1 Prior samples + analytic kernel profiles

- **Figure 6 (RBF / Matern-3/2 / RQ)**
```python
kernels_for_sampling = [
    ("RBF",    RBF(length_scale=ell)),
    ("Matern", Matern(length_scale=ell, nu=nu)),
    ("RQ",     RationalQuadratic(length_scale=ell, alpha=alpha)),
]
Y = gp_prior_samples(kernel, x_left, n_samples, rng)

def k_rbf(r, ell=1.0, sigma2=1.0):
    return sigma2 * np.exp(-(r**2)/(2.0*ell**2))
def k_matern_15(r, ell=1.0, sigma2=1.0):
    a = np.sqrt(3.0) * (np.abs(r)/ell)
    return sigma2 * (1.0 + a) * np.exp(-a)
def k_rq(r, ell=1.0, alpha=2.0, sigma2=1.0):
    return sigma2 * (1.0 + (r**2)/(2.0*alpha*ell**2))**(-alpha)
```

- **Figure 7 (PER / LIN)**  
```python
kernels_for_sampling = [
    ("PER", ExpSineSquared(length_scale=ell, periodicity=p)),
    ("LIN", DotProduct(sigma_0=sigma0)),
]
def k_per(r, ell=1.0, p=1.0, sigma2=1.0):
    return sigma2 * np.exp(-2.0 * (np.sin(np.pi*r/p)**2) / ell**2)
def k_lin(r, c=0.0, sigma2=1.0):
    return sigma2 * (r + c)
```

#### 2.2 Full GP pipelines

All figures consist of four panels: (i) noisy observations with train/test split, (ii) prior samples, (iii) posterior samples and (iv) predictive mean with 95% CI.

- **Figure 8 (RBF)**
```python
Sigma_prior = calcSigma(x_star, x_star, theta)
prior_samples = multivariate_normal.rvs(
    mean=np.zeros(len(x_star)),
    cov=Sigma_prior + 1e-6*np.eye(len(x_star)),
    size=5
)
posterior_samples = multivariate_normal.rvs(
    mean=f_star_mean,
    cov=cov_f_star + 1e-6*np.eye(len(x_star)),
    size=5
)
```

- **Figure 9 (Rational Quadratic)** – same structure as Fig. 8 with `calcSigma_RQ`  
- **Figure 10 (Matern ν=1.5)** – same structure as Fig. 8 with `calcSigma_Matern`  
- **Figure 11 (Periodic)** – same structure as Fig. 8 with `calcSigma_PER`  
- **Figure 12 (Linear)** – same structure as Fig. 8 with `calcSigma_LIN`  

- **Figures 13–18 (RQ with different θ)** – same structure as Fig. 9, but with different hyperparameters.

---

### 3) `Results of GPR.py` — Forecasting weekly egg prices

Data: sliding window N=6, last 53 obs as test set.

- **Figure 23 (RBF)**
```python
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, ...)
gp.fit(X_train, y_train)
y_pred, y_std = gp.predict(X_test, return_std=True)
```

- **Figure 24 (RQ)** – same structure with `RationalQuadratic`  
- **Figure 25 (Matern)** – same structure with `Matern`  
- **Figure 26 (Composite)** – same structure with composite kernel `RBF + RQ * DotProduct`  

- **Figure 27 (Comparison of GPR vs SARIMA)**
```python
sarima_model = SARIMAX(train_ts, order=(1,1,1), seasonal_order=(1,0,1,52))
sarima_result = sarima_model.fit(disp=False)
sarima_pred = sarima_result.get_forecast(steps=53).predicted_mean

kernel = ConstantKernel(1.0, (1e-3, 1e3)) * (
    RBF(length_scale=1.0)
    + RationalQuadratic(length_scale=1.0, alpha=1.0) * DotProduct()
)
gp = GaussianProcessRegressor(kernel=kernel, ...)
gp.fit(X_train, y_train)
y_pred_gpr, y_std_gpr = gp.predict(X_test, return_std=True)
```

---

### 4) `Gaussian Classification.py` — Trend classification (GPC)

#### 4.1 Synthetic example

- **Figure 28**
```python
f_true = 0.5 * X.ravel() + np.sin(X).ravel() + 0.3*np.cos(3*X).ravel()
p_true = 1.0 / (1.0 + np.exp(-f_true))
y = np.random.binomial(1, p_true)
kernel = ConstantKernel(1.0) * RBF(length_scale=1.5)
gpc = GaussianProcessClassifier(kernel=kernel)
```

#### 4.2 Egg-price directional movement

- **Figure 30 (RBF)**  
```python
def make_kernel(jitter):
    return ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=jitter)
kernel = make_kernel(1e-6)
gpc = GaussianProcessClassifier(kernel=kernel, max_iter_predict=300)
y_prob = gpc.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)
```

- **Figure 31 (RQ)** – same as Fig. 30 with `RationalQuadratic`  
- **Figure 32 (Matern)** – same as Fig. 30 with `Matern(ν=1.5)`  
- **Figure 33 (Linear)** – same as Fig. 30 with `DotProduct`  
- **Figure 34 (LIN+RBF composite)**  
```python
kernel = ConstantKernel(1.0) * (DotProduct(sigma_0=1.0) + RBF(length_scale=3.0))
```

- **Figure 35 (PER×RBF composite)**  
```python
def make_kernel(jitter):
    return ConstantKernel(1.0) * (
        ExpSineSquared(length_scale=1.0, periodicity=12.0)
        * RBF(length_scale=1.0)
    ) + WhiteKernel(noise_level=jitter)
```

- **Figure 36 (Matern+RBF composite)**  
```python
def make_kernel(jitter):
    return ConstantKernel(1.0) * (
        Matern(length_scale=1.0, nu=1.5)
        + RBF(length_scale=1.0)
    ) + WhiteKernel(noise_level=jitter)
```
