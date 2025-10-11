# 🧠 ML Math Lab — From Linear Algebra to Machine Learning (Python 3.13)

### 🎯 Overview
This repo is a hands-on **math-driven introduction to machine learning** — built line-by-line in Python 3.13 using PyCharm.  
Each project builds intuition by coding fundamental algorithms from scratch instead of relying only on scikit-learn.

You’ll touch:
- Linear algebra (power iteration, eigenvalues)
- Optimization (gradient descent)
- Regression (OLS, logistic)
- Unsupervised learning (k-means, PCA)

All examples run on the latest stable stack (NumPy 2.1 + SciPy 1.14 + pandas 2.2 + scikit-learn 1.6).

---

### ⚙️ Environment Setup
```bash
# clone the repo
git clone https://github.com/<your-username>/ml-math-lab.git
cd ml-math-lab

# create and activate venv (Windows example)
python -m venv .venv
.venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

requirements.txt 
numpy>=2.1
scipy>=1.14.1
pandas>=2.2.3
matplotlib>=3.9
scikit-learn>=1.6

Project structure
ml-math-lab/
├── requirements.txt
├── README.md
└── projects/
    ├── 1_power_iteration.py
    ├── 2_ols_from_scratch.py
    ├── 3_logreg_scratch.py
    ├── 4_kmeans_scratch.py
    └── 5_pca_svd.py
🚀 Projects Summary
1️⃣ Power Iteration — Dominant Eigenvalue & Eigenvector

Finds the main direction a matrix stretches space.

Foundation for PCA, PageRank, and factor models.

Output: top eigenvalue λ* and eigenvector v*.

Example:

Power iteration λ* : 13.929951
NumPy top λ*       : 13.929951
Angle difference (rad): 0.0
2️⃣ Ordinary Least Squares (OLS) Regression

Derives the closed-form solution 
𝛽
^
=
(
𝑋
𝑇
𝑋
)
−
1
𝑋
𝑇
𝑦
β
^
	​

=(X
T
X)
−1
X
T
y.

Teaches matrix algebra behind linear regression.

Compared to sklearn.LinearRegression for verification.

3️⃣ Logistic Regression (Gradient Descent)

Implements binary classification using sigmoid activation.

Updates weights via gradient descent; visualizes loss convergence.

Reinforces optimization and convex loss minimization concepts.

4️⃣ k-Means Clustering

From-scratch EM-style algorithm for unsupervised grouping.

Shows centroid updates, assignments, and inertia minimization.

Compares against sklearn.KMeans for validation.

5️⃣ PCA via SVD

Performs dimensionality reduction through singular value decomposition.

Extracts top-k components, explained variance ratios, and reconstruction error.

Direct bridge between linear algebra and ML preprocessing.

📊 Core Skills You’ll Strengthen
Category	Skill
Linear Algebra	Eigenvalues, SVD, orthogonality
Optimization	Gradient descent, residual minimization
Statistics	Regression, variance, covariance
Unsupervised Learning	Clustering, dimensionality reduction
Python	NumPy, pandas, scikit-learn, vectorized math
💡 Why This Repo Exists

“If you can build the math from scratch, you can master any ML library.”

This project is designed for finance, quant research, and ML learners who want deep intuition — not black-box modeling.
🧠 Credits

Created by Clyde Williams Jr.
Built entirely in PyCharm using Python 3.13 for educational and quant-finance preparation.

Connect

LinkedIn: linkedin.com/in/clydewilliamsjr

GitHub: github.com/<QuantTrader0317>