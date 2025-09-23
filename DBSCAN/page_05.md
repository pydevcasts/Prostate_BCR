## 📖 فصل پنجم: اجرای K-Means روی Moons و Circles

قبل از اجرای DBSCAN، بیایم ببینیم که چرا الگوریتم K-Means در این نوع داده‌ها به مشکل می‌خوره.

---

### 📌 کد: اجرای K-Means

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Run KMeans on moons dataset
kmeans_moons = KMeans(n_clusters=2, random_state=42)
y_kmeans_moons = kmeans_moons.fit_predict(X_moons)

# Run KMeans on circles dataset
kmeans_circles = KMeans(n_clusters=2, random_state=42)
y_kmeans_circles = kmeans_circles.fit_predict(X_circles)

# Plotting results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Moons
axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_kmeans_moons, cmap="viridis", s=30)
axes[0].scatter(kmeans_moons.cluster_centers_[:, 0], kmeans_moons.cluster_centers_[:, 1], 
                c="red", marker="x", s=200, label="Centers")
axes[0].set_title("K-Means on Moons")
axes[0].legend()

# Circles
axes[1].scatter(X_circles[:, 0], X_circles[:, 1], c=y_kmeans_circles, cmap="viridis", s=30)
axes[1].scatter(kmeans_circles.cluster_centers_[:, 0], kmeans_circles.cluster_centers_[:, 1], 
                c="red", marker="x", s=200, label="Centers")
axes[1].set_title("K-Means on Circles")
axes[1].legend()

plt.show()
```

---

### 🔍 تحلیل نتایج

* در دیتاست **moons**:
  خوشه‌بندی درست انجام نشده و نقاطی از یک هلال به اشتباه در خوشه دیگر قرار گرفتن. دلیلش اینه که K-Means فرض می‌کنه خوشه‌ها باید کروی باشن.

* در دیتاست **circles**:
  K-Means دایره‌های تو در تو رو اشتباه دسته‌بندی می‌کنه چون بر اساس فاصله از مرکز دسته‌بندی می‌کنه و ساختار دایره‌ای رو درک نمی‌کنه.

