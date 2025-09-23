## 🔹 کد پیاده‌سازی یک تکرار EM برای GMM (به‌صورت دستی)

```python
import numpy as np
from scipy.stats import norm

# Data
X = np.array([1, 2, 3, 8, 9, 10])

# Initial parameters
pi = np.array([0.5, 0.5])   # Mixing coefficients
mu = np.array([2.0, 9.0])   # Means
sigma = np.array([1.0, 1.0])  # Standard deviations

# ---------- E-step ----------
# Compute responsibilities gamma_{ik}
responsibilities = np.zeros((len(X), 2))

for i, x in enumerate(X):
    # Compute probability density for each component
    p1 = pi[0] * norm.pdf(x, mu[0], sigma[0])
    p2 = pi[1] * norm.pdf(x, mu[1], sigma[1])
    # Normalize to sum=1
    responsibilities[i, 0] = p1 / (p1 + p2)
    responsibilities[i, 1] = p2 / (p1 + p2)

print("Responsibilities (gamma_ik):\n", responsibilities)

# ---------- M-step ----------
N_k = responsibilities.sum(axis=0)  # Effective number of points per cluster
mu_new = (responsibilities * X[:, np.newaxis]).sum(axis=0) / N_k
sigma_new = np.sqrt(((responsibilities * (X[:, np.newaxis] - mu_new) ** 2).sum(axis=0)) / N_k)
pi_new = N_k / len(X)

print("\nUpdated parameters after 1 EM iteration:")
print("Mu:", mu_new)
print("Sigma:", sigma_new)
print("Pi:", pi_new)
```

📌 این کد همون چیزی رو که دستی محاسبه کردیم به‌صورت خودکار انجام می‌ده.

---

## 🔹 استفاده از کتابخانه‌ی `GaussianMixture` برای چندین تکرار

حالا همون داده‌ها رو به `sklearn` می‌دیم تا ببینیم بعد از چند تکرار چه نتایجی به‌دست می‌آید:

```python
from sklearn.mixture import GaussianMixture

# Fit GMM with 2 components
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X.reshape(-1, 1))

print("Final parameters (after convergence):")
print("Means:", gmm.means_.ravel())
print("Variances:", gmm.covariances_.ravel())
print("Weights:", gmm.weights_)
print("Responsibilities:\n", gmm.predict_proba(X.reshape(-1, 1)))
```

---

## 🔹 مصورسازی

برای اینکه ببینیم GMM داده‌ها رو چطور مدل کرده، یک نمودار توزیع احتمال مخلوط رسم می‌کنیم:

```python
import matplotlib.pyplot as plt

# Create a grid of points for plotting
x_grid = np.linspace(0, 11, 500)
dens = np.exp(gmm.score_samples(x_grid.reshape(-1, 1)))

# Densities of individual Gaussians
dens_individual = np.array([
    w * norm.pdf(x_grid, mu, np.sqrt(var))
    for w, mu, var in zip(gmm.weights_, gmm.means_.ravel(), gmm.covariances_.ravel())
])

plt.figure(figsize=(8, 5))
plt.hist(X, bins=20, density=True, alpha=0.5, color="gray", label="Data histogram")
plt.plot(x_grid, dens, label="GMM total density", color="red", linewidth=2)

for i, d in enumerate(dens_individual):
    plt.plot(x_grid, d, "--", label=f"Component {i+1}")

plt.scatter(X, [0]*len(X), marker="x", color="black", zorder=10, label="Data points")
plt.title("Gaussian Mixture Model on simple dataset")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.show()
```

---

## 🔹 نتیجه

* GMM دقیقاً داده‌ها را به دو خوشه‌ی جداگانه (اطراف ۲ و ۹) تقسیم می‌کند.
* هر داده به‌جای اختصاص سخت (k-Means) یک **احتمال تعلق** به هر خوشه دارد.
* منحنی‌های گاوسی جداگانه و همچنین ترکیب آن‌ها (مخلوط گاوسی کلی) روی نمودار دیده می‌شوند.

---

📘 پیشنهاد: می‌خوای برای فصل کتاب، همین مثال رو هم با **تحلیل تصویری نتایج مسئولیت‌ها (barplot یا heatmap)** اضافه کنیم؟ تا دانشجوها بهتر درک کنن که چطور احتمال تعلق هر نقطه بین خوشه‌ها تقسیم میشه.
