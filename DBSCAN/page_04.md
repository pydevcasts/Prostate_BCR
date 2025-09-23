## 📖 فصل چهارم: ایجاد دیتاست‌های مصنوعی برای DBSCAN

برای اینکه قدرت الگوریتم DBSCAN رو بهتر درک کنیم، از دیتاست‌های مصنوعی استفاده می‌کنیم که شکل‌های غیرخطی دارن. معروف‌ترین این دیتاست‌ها:

1. **Moons (دو هلال)**

   * داده‌ها به شکل دو نیم‌دایره در کنار هم هستن.
   * K-Means معمولاً نمی‌تونه اون‌ها رو درست دسته‌بندی کنه.

2. **Circles (دایره‌های تو در تو)**

   * یک دایره کوچیک درون یک دایره بزرگ‌تر.
   * K-Means نمی‌تونه این ساختار رو درست شناسایی کنه چون خوشه‌ها کروی و متمرکز نیستن.

---

### 📌 کد: ساخت و نمایش دیتاست‌های Moons و Circles

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles

# Create moons dataset
X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)

# Create circles dataset
X_circles, y_circles = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)

# Plot both datasets
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Moons
axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap="viridis", s=30)
axes[0].set_title("Moons Dataset")

# Plot Circles
axes[1].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap="viridis", s=30)
axes[1].set_title("Circles Dataset")

plt.show()
```

---

### 🔍 تحلیل نمودارها

* در **moons** می‌بینیم که داده‌ها دو خوشه هلالی شکل دارن. این ساختار برای K-Means مشکل ایجاد می‌کنه چون اون الگوریتم فقط خوشه‌های کروی رو خوب شناسایی می‌کنه.
* در **circles** داده‌ها به شکل دو دایره تو در تو هستن. K-Means این‌ها رو اشتباه ترکیب می‌کنه چون فاصله از مرکز رو معیار قرار می‌ده.


