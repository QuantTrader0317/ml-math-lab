# 🧮 Power Iteration — Finding the Dominant Eigenvalue & Eigenvector

### 🔍 What It Does
This script implements the **Power Iteration** algorithm — a simple numerical method to find the **dominant eigenvalue** (λ\*) and **eigenvector** of any square matrix *A*.  

In plain English:  
> It discovers the direction where a matrix stretches space the most, and measures how strong that stretch is.

---

### ⚙️ How It Works
1. Start with a random vector *x₀*.  
2. Multiply repeatedly: *xₖ₊₁ = A · xₖ / ‖A · xₖ‖*.  
3. After each step, estimate the stretch (λ) with the **Rayleigh quotient** λ = xᵀ A x.  
4. Stop when the **residual** ‖A x − λ x‖ is tiny (< tolerance).  
5. The final λ and x are the top eigenpair.

---

### 🧠 Intuition
- **A** acts like a transformation machine.  
- **x** is a direction in space.  
- Repeatedly applying A pulls x toward the direction A naturally stretches the most.  
- Once the direction stops changing, you’ve found that dominant axis.

---

### 🧩 Example Output
Power iteration λ* : 13.929951
NumPy top λ* : 13.929951
Angle difference (radians): 0.0

✅ λ\* matches NumPy’s result → the algorithm converged.  
✅ Angle ≈ 0 → your vector perfectly aligns with the true eigenvector.

---

### 🧠 Why It Matters
| Field | What It Represents |
|:--|:--|
| **Quant Finance** | Dominant risk factor / largest eigenvalue of a covariance matrix |
| **Machine Learning (PCA)** | First principal component |
| **Search Engines** | Mathematical basis of PageRank |
| **Physics / Control** | System stability mode |

---

