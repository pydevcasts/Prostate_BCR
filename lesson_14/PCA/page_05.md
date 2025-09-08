
---

## 📖 صفحه ۵: محاسبه ماتریس کوواریانس و تجزیه ویژه (Eigen Decomposition)

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 استانداردسازی داده‌ها

پیش از محاسبه ماتریس کوواریانس، باید داده‌ها را استاندارد کنیم. چون برخی ویژگی‌ها مثل **Alcohol** بین ۱۲ تا ۱۵ هستند ولی ویژگی **Magnesium** مقادیر کوچکی دارد. اگر داده‌ها استاندارد نشوند، ویژگی‌هایی با مقیاس بزرگ‌تر بر نتایج PCA غالب می‌شوند.

فرمول استانداردسازی:

$$
X_{scaled} = \frac{X - \mu}{\sigma}
$$

---

### 🔹 محاسبه ماتریس کوواریانس

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.drop('target', axis=1))

# Compute covariance matrix
cov_matrix = np.cov(X_scaled.T)

print("Shape of covariance matrix:", cov_matrix.shape)
```

📌 تحلیل:

* شکل ماتریس کوواریانس `(13, 13)` خواهد بود چون ۱۳ ویژگی داریم.
* هر مقدار $C_{ij}$ نشان‌دهنده‌ی میزان همبستگی بین ویژگی i و j است.

---

### 🔹 محاسبه Eigenvalues و Eigenvectors

```python
# Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Eigenvalues:\n", eigenvalues)
print("\nEigenvectors shape:", eigenvectors.shape)
```

📌 تحلیل:

* **Eigenvalues** نشان‌دهنده اهمیت هر مؤلفه هستند. هر چه مقدار بزرگ‌تر باشد، آن مؤلفه واریانس بیشتری از داده را توضیح می‌دهد.
* **Eigenvectors** بردارهای جهت هستند که فضای جدید PCA را تعریف می‌کنند.

---

### 🔹 مرتب‌سازی مؤلفه‌ها

برای انتخاب مؤلفه‌های اصلی، Eigenvalues را از بزرگ به کوچک مرتب می‌کنیم.

```python
# Sort eigenvalues and eigenvectors
sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

print("Sorted Eigenvalues:\n", eigenvalues)
```

📌 تحلیل:

* بزرگ‌ترین مقدار ویژه (λ1) مربوط به **PC1** است.
* دومین مقدار ویژه (λ2) مربوط به **PC2** است.
* همین‌طور ادامه دارد تا PC13.

---

### 🔹 نتیجه‌گیری این گام

1. ماتریس کوواریانس نشان داد که برخی ویژگی‌ها همبستگی بالایی دارند.
2. Eigenvalues و Eigenvectors محاسبه شدند تا جهت مؤلفه‌های اصلی مشخص شوند.
3. مؤلفه‌هایی که Eigenvalue بزرگ‌تری دارند، بیشترین واریانس داده را توضیح می‌دهند و برای کاهش بعد انتخاب می‌شوند.
