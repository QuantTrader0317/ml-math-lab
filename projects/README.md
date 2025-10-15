# 🧠 Machine Learning Math Lab — 6 Core Projects in Python (Built from Scratch)

### 🚀 Overview
This project is my hands-on lab for mastering the **mathematical foundations of machine learning** using pure Python (v3.13) and NumPy.

Everything here was written, debugged, and explained line by line inside **PyCharm**, to fully understand *how the math behind each model actually works* — not just how to call it from a library.

---

## 📂 Projects Included

| # | Project | Core Concept | Key Skill |
|:-:|:--|:--|:--|
| 1 | **Power Iteration** | Eigenvalues & Eigenvectors | Linear Algebra Fundamentals |
| 2 | **Ordinary Least Squares (OLS)** | Linear Regression | Matrix Inversion & R² Analysis |
| 3 | **Logistic Regression (Gradient Descent)** | Classification | Optimization & Cost Functions |
| 4 | **k-Means Clustering** | Unsupervised Learning | Distance Metrics & Iterative Refinement |
| 5 | **Principal Component Analysis (PCA)** | Dimensionality Reduction | SVD, Variance, & Reconstruction |
| 6 | **Regularized Regression (Ridge & Lasso)** | Overfitting Control | L2 & L1 Penalties, Feature Selection |
| 7 | **Optimizer Comparison (SGD • Momentum • Adam)** | Optimization Algorithms | Convergence Speed & Stability |
| 8 | **Time-Series Analysis (Stationarity & Autocorrelation)** | Statistical Properties of Market Data | ADF Tests, ACF, Ljung–Box, AR(1) Model |
| 9 | **Feature Engineering for Returns** | Quant Signal Design | Momentum, Volatility, Z-Scores, Rolling Beta, Lagged Returns |

---

### 💡 What I Learned
- How to **translate math formulas into working code**  
- Why each algorithm converges (or fails)  
- The difference between **exact math** and **numerical stability**  
- How modern libraries like scikit-learn are built under the hood  
- How **regularization stabilizes models** and performs **automatic feature selection**

---

### ⚙️ Tools Used
- Python 3.13  
- NumPy 2.1+  
- scikit-learn (for cross-verification)  
- PyCharm (for daily environment setup, debugging, and testing)

---

### 🧩 Example Insights
- **Power Iteration:** a matrix naturally “pulls” vectors toward its strongest direction  
- **OLS Regression:** fitting a line is just solving `XᵀXβ = Xᵀy`  
- **Logistic Regression:** prediction is probability, not just classification  
- **k-Means:** patterns form through repeated nearest-center updates  
- **PCA:** you can compress data while keeping most of its meaning  
- **Ridge & Lasso:** adding penalties keeps models stable and highlights the most important features  
- **Optimizers (SGD vs Momentum vs Adam):** All train the same logistic model differently.  
  - SGD → simple but noisy steps  
  - Momentum → adds velocity for smoother progress  
  - Adam → adapts learning rate per weight, usually fastest and most stable  
  These control *how models actually learn*, which is crucial for larger ML & quant research systems.
- **Time-Series Analysis:** Prices are non-stationary (random-walk behavior); returns are stationary.  
  Weak negative autocorrelation indicates short-term mean reversion — a foundation for quant forecasting.  
- **Feature Engineering for Returns:** Built momentum, volatility, z-score, beta, and lagged-return features.  
  Individually weak but collectively form the basis of a multi-factor predictive signal.

---

### 🏁 Key Takeaway
> “If you can code the math from scratch, you can understand any ML model — no black boxes.”

This repo built my foundation for deeper research in **quantitative finance, econometrics, and machine learning.**

---

### 📸 Optional Demo Post
> Just finished building 6 core ML algorithms completely from scratch in Python.  
> No high-level wrappers, no shortcuts — just math, NumPy, and logic.  
> The goal wasn’t just to get the right answer, but to *understand why* it’s right.  
>
> 🧮 Power Iteration → Eigenvalues  
> 📈 OLS Regression → Linear Models  
> 🔁 Logistic Regression → Optimization  
> 🎯 k-Means → Clustering  
> 🔍 PCA → Dimensionality Reduction  
> ⚖️ Ridge & Lasso → Regularization & Feature Selection  
>
> All coded and explained inside PyCharm using Python 3.13.  
> This is how you turn theory into intuition.

---

### 📁 Folder Layout
ml-math-lab/
│
├── projects/
│ ├── 1_power_iteration.py
│ ├── 2_ols_from_scratch.py
│ ├── 3_logreg_scratch.py
│ ├── 4_kmeans_scratch.py
│ ├── 5_pca_svd.py
│ └── 6_regularized_regression.py
│ └── 7_optimizers_logreg.py
│ └── 8_time_series_basics.py
│ └── 9_feature_engineering_returns.py
├── README.md ← this file
└── requirements.txt

---

### ✅ Next Step Ideas
- **Project 7:** Optimizers (SGD, Momentum, Adam)  
- **Project 8:** Time-Series Basics for Returns  
- **Project 9:** Feature Engineering for Quant Signals  
- **Project 10:** Backtester with Transaction Costs  
- **Project 11:** Portfolio Construction (Ridge/Volatility Targeting)  
- **Project 12:** Broker API Integration (Paper Trading)  
- **Project 13:** Risk & Monitoring Dashboard  

---

**Created by:** Clyde Williams Jr.  
**Focus:** Quantitative Finance • Machine Learning • Mathematical Research  
**Stack:** Python | NumPy | Statistics | Linear Algebra
