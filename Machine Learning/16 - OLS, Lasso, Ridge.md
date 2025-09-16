# 📘 Notes on OLS, Lasso, and Ridge Regression

This document summarizes the key concepts of **Ordinary Least Squares (OLS)**, **Lasso Regression**, and **Ridge Regression**.  
It is meant as a quick reference for machine learning workflows.

---

## 1. Ordinary Least Squares (OLS) – Linear Regression
**Definition**  
OLS is the basic form of linear regression. It fits a line/plane (in higher dimensions) to minimize the **sum of squared errors (SSE)** between actual and predicted values.

**Objective Function:**
\[
\min_\beta \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

**Characteristics:**
- Uses all available features.  
- Sensitive to **outliers** and **multicollinearity**.  
- No built-in regularization (risk of overfitting if features are many).  

**When to use:**
- When data is small, clean, and not highly correlated.  

---

## 2. Ridge Regression – L2 Regularization
**Definition**  
Ridge adds an **L2 penalty** to OLS, shrinking coefficients towards zero but **never eliminating them**.

**Objective Function:**
\[
\min_\beta \left( \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2 \right)
\]

**Characteristics:**
- Shrinks all coefficients proportionally.  
- Keeps all features (no true feature elimination).  
- Works well when many features are relevant but with small contributions.  
- Reduces variance and handles **multicollinearity** better.  

**When to use:**
- High-dimensional datasets with correlated features.  
- When you want to keep all predictors but control their impact.  

---

## 3. Lasso Regression – L1 Regularization
**Definition**  
Lasso adds an **L1 penalty** to OLS, shrinking some coefficients exactly to zero, effectively performing **feature selection**.

**Objective Function:**
\[
\min_\beta \left( \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j| \right)
\]

**Characteristics:**
- Automatically selects a subset of features.  
- Can reduce model complexity by eliminating irrelevant predictors.  
- Sensitive when features are highly correlated (may arbitrarily pick one).  

**When to use:**
- When you suspect many features are irrelevant.  
- Useful for sparse models (where only a few predictors matter).  

---

## 4. Comparison Table

| Aspect                  | OLS (Linear Regression) | Ridge (L2) | Lasso (L1) |
|--------------------------|--------------------------|------------|------------|
| Regularization           | ❌ None                 | ✅ Yes (L2) | ✅ Yes (L1) |
| Coefficient shrinkage    | ❌ No                   | ✅ Yes (all reduced, none = 0) | ✅ Yes (some = 0 → feature selection) |
| Handles multicollinearity| ❌ Poor                 | ✅ Good    | ⚠️ Mixed (may drop correlated features) |
| Feature selection        | ❌ No                   | ❌ No      | ✅ Yes |
| Typical use case         | Simple, clean data      | Keep all features but reduce variance | Sparse model with feature selection |

---

## 5. Key Intuition
- **OLS**: Fits data exactly → risk of overfitting.  
- **Ridge**: Balances → keeps all features, reduces variance.  
- **Lasso**: Enforces sparsity → removes irrelevant features.  

---

## 6. Bonus: ElasticNet (L1 + L2 mix)
Sometimes, combining Ridge and Lasso gives the best of both worlds.  
ElasticNet uses both penalties:

\[
\text{Loss} = \text{SSE} + \lambda_1 \sum |\beta_j| + \lambda_2 \sum \beta_j^2
\]

- Keeps correlated features together (unlike Lasso).  
- Provides feature selection + stability.  

---
