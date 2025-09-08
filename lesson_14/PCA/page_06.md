
---

## 📖 صفحه ۶: پیاده‌سازی PCA با NumPy و Scikit-Learn

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 روش اول: پیاده‌سازی PCA با NumPy (گام به گام)

در این بخش، همان مراحلی که در صفحه قبل توضیح دادیم را با NumPy عملی می‌کنیم.

```python
# Project data onto first 2 principal components using NumPy
X_pca_numpy = X_scaled.dot(eigenvectors[:, :2])

print("Shape of transformed data:", X_pca_numpy.shape)
```

📌 تحلیل:

* داده‌ها از فضای ۱۳ بعدی به فضای ۲ بعدی فشرده شدند.
* حالا می‌توانیم آن‌ها را روی یک نمودار دوبعدی رسم کنیم.

---

### 🔹 Visualization خروجی NumPy PCA

```python
plt.figure(figsize=(8,6))
plt.scatter(X_pca_numpy[:,0], X_pca_numpy[:,1], c=data['target'], cmap='viridis', edgecolor='k')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection (NumPy Implementation)")
plt.colorbar(label="Wine Class")
plt.show()
```

📌 مشاهده می‌کنیم که داده‌ها در فضای دوبعدی نسبتاً خوب از هم جدا شده‌اند، مخصوصاً کلاس ۰ از کلاس‌های دیگر فاصله دارد.

---

### 🔹 روش دوم: پیاده‌سازی PCA با Scikit-Learn

کتابخانه Scikit-Learn توابع آماده برای PCA دارد که کار را ساده‌تر و سریع‌تر می‌کند.

```python
from sklearn.decomposition import PCA

# Apply PCA with 2 components
pca = PCA(n_components=2)
X_pca_sklearn = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)
```

📌 خروجی:

* `explained_variance_ratio_` نشان می‌دهد هر مؤلفه چه درصدی از واریانس داده را توضیح می‌دهد.
* مثلاً اگر PC1 = 0.36 و PC2 = 0.19 باشد، یعنی در مجموع ۵۵٪ از اطلاعات داده با ۲ مؤلفه اول حفظ شده است.

---

### 🔹 Visualization خروجی Scikit-Learn

```python
plt.figure(figsize=(8,6))
plt.scatter(X_pca_sklearn[:,0], X_pca_sklearn[:,1], c=data['target'], cmap='plasma', edgecolor='k')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection (Scikit-Learn Implementation)")
plt.colorbar(label="Wine Class")
plt.show()
```

📌 نتیجه مشابه NumPy است، اما Scikit-Learn محاسبات را بهینه‌تر و همراه با اطلاعات اضافی (مثل واریانس توضیح داده‌شده) ارائه می‌دهد.

---

### 🔹 نتیجه این بخش

1. PCA را هم با NumPy و هم با Scikit-Learn پیاده‌سازی کردیم.
2. داده‌های ۱۳ بعدی به ۲ بعد فشرده شدند و همچنان جداسازی کلاس‌ها قابل مشاهده است.
3. مؤلفه‌های اصلی بیشترین واریانس داده را پوشش دادند و Visualization ساده‌تر شد.

