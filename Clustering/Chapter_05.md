# 📖 فصل پنجم: خوشه‌بندی با الگوریتم DBSCAN

---

## 🔹 چرا DBSCAN؟

الگوریتم‌های K-Means و Hierarchical معمولاً فرض می‌کنند خوشه‌ها **شکل کروی یا ساده** دارند. اما در داده‌های واقعی خوشه‌ها ممکن است شکل‌های پیچیده داشته باشند یا شامل نویز (outliers) باشند.

اینجاست که **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** وارد می‌شود.

---

## 🔹 ایده اصلی DBSCAN

* خوشه‌ها مناطقی با **تراکم بالا** از نقاط هستند.
* نقاطی که در مناطق کم‌تراکم قرار دارند، به عنوان **نویز** در نظر گرفته می‌شوند.

DBSCAN بر اساس دو پارامتر کار می‌کند:

* **ε (epsilon)**: شعاع همسایگی
* **minPts**: حداقل تعداد نقاط موردنیاز در یک همسایگی برای تشکیل خوشه

---

## 📐 دسته‌بندی نقاط در DBSCAN

1. **Core Point**: نقطه‌ای با حداقل minPts در ε همسایگی‌اش
2. **Border Point**: نقطه‌ای که همسایگی‌اش کمتر از minPts دارد، اما در ε یک Core Point قرار گرفته است
3. **Noise Point**: نقطه‌ای که هیچ‌کدام از شرایط بالا را ندارد

---

## 📊 پیاده‌سازی DBSCAN روی دیتاست Iris

```python
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Apply DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=5)
df['dbscan_cluster'] = dbscan.fit_predict(X_scaled)

# Reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['dbscan_cluster'], palette="Set1")
plt.title("DBSCAN Clustering on Iris dataset (PCA reduced)")
plt.show()
```

📌 مشاهده:

* DBSCAN توانسته برخی خوشه‌ها را به خوبی جدا کند.
* بعضی نقاط به عنوان **نویز** (label = -1) شناسایی شده‌اند.

---

## 📊 مقایسه با برچسب‌های واقعی

```python
pd.crosstab(df['target'], df['dbscan_cluster'])
```

📌 نتیجه نشان می‌دهد که:

* کلاس Setosa باز هم به خوبی جدا شده است.
* بین Versicolor و Virginica همپوشانی وجود دارد، که طبیعی است چون این دو کلاس حتی در داده‌های اصلی هم مرز مشخصی ندارند.

---

## ✨ مزایا و معایب DBSCAN

✅ مزایا:

* نیاز به تعیین تعداد خوشه‌ها از قبل ندارد.
* توانایی شناسایی خوشه‌های با اشکال پیچیده.
* شناسایی نویز و outliers.

❌ معایب:

* انتخاب پارامترهای ε و minPts دشوار است.
* روی داده‌های با ابعاد بالا عملکرد ضعیف‌تری دارد.
