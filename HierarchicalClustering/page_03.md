## کد پایتون 📌

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

# تعریف نقاط
points = np.array([
    [0, 0],  # A
    [0, 1],  # B
    [1, 0],  # C
    [5, 5]   # D
])

labels = ["A", "B", "C", "D"]

# محاسبه ماتریس فاصله
dist_matrix = squareform(pdist(points, metric="euclidean"))
print("Distance matrix:\n", dist_matrix)

# سه روش linkage
methods = ["single", "complete", "average"]

plt.figure(figsize=(15, 5))
for i, method in enumerate(methods, 1):
    Z = linkage(points, method=method, metric="euclidean")
    
    plt.subplot(1, 3, i)
    dendrogram(Z, labels=labels)
    plt.title(f"{method.capitalize()} linkage")

plt.tight_layout()
plt.show()
```

---

## خروجی 📊

1. **ماتریس فاصله** همان جدولی است که دستی محاسبه کردیم:

   ```
   [[0.    1.    1.    7.071]
    [1.    0.    1.414 6.403]
    [1.    1.414 0.    6.403]
    [7.071 6.403 6.403 0.   ]]
   ```

2. **سه نمودار دندروگرام** کنار هم نمایش داده می‌شوند:

   * در **single linkage**، ادغام A و B و سپس C خیلی زود و در ارتفاع ۱ اتفاق می‌افتد.
   * در **complete linkage**، ادغام C کمی دیرتر و در ارتفاع ۱.۴۱۴ رخ می‌دهد.
   * در **average linkage**، این ادغام در ارتفاع میانی (۱.۲۰۷) انجام می‌شود.

---

این نمودارها دقیقاً همان ارتفاع‌هایی را که دستی حساب کردیم نشان می‌دهند ✅ و دانشجو می‌تواند تأثیر روش linkage بر شکل درخت و ترتیب ادغام‌ها را ببیند.

