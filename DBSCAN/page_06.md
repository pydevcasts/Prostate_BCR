## 📖 فصل ششم: اجرای DBSCAN روی دیتاست Moons

الگوریتم DBSCAN با دو پارامتر اصلی کنترل می‌شه:

* $\varepsilon$ (Epsilon): شعاع همسایگی
* **minPts**: حداقل تعداد نقاط موردنیاز در همسایگی برای تشکیل یک **Core Point**

---

### 📌 کد: اجرای DBSCAN روی Moons

```python
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Run DBSCAN on moons dataset
dbscan = DBSCAN(eps=0.2, min_samples=5)  
y_dbscan_moons = dbscan.fit_predict(X_moons)

# Plot DBSCAN results
plt.figure(figsize=(6, 5))
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_dbscan_moons, cmap="plasma", s=30)
plt.title("DBSCAN on Moons")
plt.show()
```

---

### 🔍 تحلیل نتایج

* می‌بینیم که DBSCAN دو هلال رو **به‌خوبی جدا کرده**.
* برخلاف K-Means، اینجا الگوریتم نیازی به فرض کروی بودن خوشه‌ها نداره و بر اساس **چگالی نقاط** خوشه‌ها رو شناسایی می‌کنه.
* اگر بعضی نقاط در دسته‌بندی قرار نگیرن، DBSCAN اون‌ها رو به‌عنوان **نویز (Noise)** مشخص می‌کنه (معمولاً با برچسب `-1`).

