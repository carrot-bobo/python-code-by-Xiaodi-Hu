# Gaussian Process Regression and Classification for Egg Price Forecasting

This repository contains Python implementations of **Gaussian Process Regression** and **Gaussian Process Classification** applied to agricultural commodity price forecasting, with a focus on egg prices in China.  
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


