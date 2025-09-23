# 📖 فصل هفتم: اجرای DBSCAN روی دیتاست Circles

### 📌 کد تولید دیتاست و اجرای DBSCAN

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN

# Generate circles dataset
X_circles, y_circles = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)

# Run DBSCAN
dbscan_circles = DBSCAN(eps=0.1, min_samples=5)
y_dbscan_circles = dbscan_circles.fit_predict(X_circles)

# Plot DBSCAN results
plt.figure(figsize=(6, 5))
plt.scatter(X_circles[:, 0], X_circles[:, 1], c=y_dbscan_circles, cmap="plasma", s=30)
plt.title("DBSCAN on Circles")
plt.show()
```

---

### 🔍 تحلیل نتایج

* الگوریتم DBSCAN دو دایره تو در تو رو **به‌درستی شناسایی کرده**.
* نقاطی که در دسته‌بندی قرار نگرفتن با برچسب `-1` به عنوان **نویز** مشخص شدن.
* برتری DBSCAN نسبت به K-Means اینجا خیلی روشنه:

  * K-Means به دلیل فرض کروی بودن خوشه‌ها نمی‌تونه دایره‌های تو در تو رو درست جدا کنه.
  * DBSCAN چون بر اساس **چگالی نقاط** عمل می‌کنه، می‌تونه این ساختار پیچیده رو شناسایی کنه.


