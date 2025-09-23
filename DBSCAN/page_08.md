
# 📖 فصل هشتم: اجرای DBSCAN روی دیتاست Circles (تکمیلی)

---

## ۱. اثر تغییر پارامترها روی خروجی DBSCAN

DBSCAN دو پارامتر اصلی داره:

* $\varepsilon$ یا **eps** → شعاع همسایگی
* **min\_samples** → حداقل تعداد نقاط لازم در همسایگی

📌 اگر مقدار **eps** خیلی کوچک باشه، الگوریتم نقاط کمی رو همسایه در نظر می‌گیره و بیشتر نقاط به عنوان **نویز (-1)** علامت می‌خورن.
📌 اگر مقدار **eps** خیلی بزرگ باشه، خوشه‌ها به هم می‌چسبن و الگوریتم ممکنه همه داده‌ها رو یک خوشه در نظر بگیره.

---

### 📌 کد: تست پارامتر eps

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Small eps (too strict)
db1 = DBSCAN(eps=0.05, min_samples=5).fit_predict(X_circles)
axes[0].scatter(X_circles[:, 0], X_circles[:, 1], c=db1, cmap="plasma", s=30)
axes[0].set_title("DBSCAN - eps=0.05")

# Medium eps (good choice)
db2 = DBSCAN(eps=0.1, min_samples=5).fit_predict(X_circles)
axes[1].scatter(X_circles[:, 0], X_circles[:, 1], c=db2, cmap="plasma", s=30)
axes[1].set_title("DBSCAN - eps=0.1")

# Large eps (too loose)
db3 = DBSCAN(eps=0.3, min_samples=5).fit_predict(X_circles)
axes[2].scatter(X_circles[:, 0], X_circles[:, 1], c=db3, cmap="plasma", s=30)
axes[2].set_title("DBSCAN - eps=0.3")

plt.show()
```

---

### 🔍 تحلیل نتایج تغییر eps

* در **eps=0.05** بیشتر نقاط به عنوان نویز برچسب خوردن.
* در **eps=0.1** الگوریتم بهترین نتیجه رو داده و دو دایره جدا شدن.
* در **eps=0.3** خوشه‌ها به هم چسبیدن و عملاً الگوریتم یک خوشه بزرگ ساخته.

این نشون میده انتخاب درست پارامترها در DBSCAN خیلی حیاتی هست.

---

## ۲. مقایسه K-Means و DBSCAN روی Circles

### 📌 کد مقایسه

```python
from sklearn.cluster import KMeans

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# K-Means on circles
kmeans_circles = KMeans(n_clusters=2, random_state=42).fit_predict(X_circles)
axes[0].scatter(X_circles[:, 0], X_circles[:, 1], c=kmeans_circles, cmap="plasma", s=30)
axes[0].set_title("K-Means on Circles")

# DBSCAN on circles
dbscan_circles = DBSCAN(eps=0.1, min_samples=5).fit_predict(X_circles)
axes[1].scatter(X_circles[:, 0], X_circles[:, 1], c=dbscan_circles, cmap="plasma", s=30)
axes[1].set_title("DBSCAN on Circles")

plt.show()
```

---

### 🔍 تحلیل مقایسه

* **K-Means** دایره‌ها رو به درستی تشخیص نمی‌ده و به‌جای دو حلقه، نقاط رو به دو نیمه تقسیم می‌کنه.
* **DBSCAN** به دلیل چگالی‌محور بودن، حلقه‌ها رو کامل و طبیعی تشخیص می‌ده.

---

## ۳. نقش نقاط نویز در DBSCAN

یکی از ویژگی‌های جالب DBSCAN، شناسایی نقاطی هست که به هیچ خوشه‌ای تعلق ندارن.
این نقاط معمولاً با برچسب `-1` نمایش داده می‌شن.

📌 در عمل، این قابلیت برای:

* تشخیص داده‌های غیرعادی (Anomaly Detection)
* حذف داده‌های پرت (Outliers)

خیلی مهم و کاربردیه.

