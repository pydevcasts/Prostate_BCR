# 📖 مرحله ششم: انتخاب تعداد خوشه بهینه

در K-Means باید مقدار **k** (تعداد خوشه‌ها) رو مشخص کنیم.
دو روش پرکاربرد برای انتخاب k:

1. **روش Elbow**
2. **روش Silhouette Analysis**

---

## 🔹 ۱. روش Elbow

در این روش، **Inertia** یا همان مجموع مربعات فاصله داده‌ها از مراکز خوشه‌ها برای مقادیر مختلف k محاسبه می‌شود.
هر چه k بیشتر شود، Inertia کمتر می‌شود، اما از یک نقطه به بعد کاهش آن کند می‌شود. آن نقطه "آرنج" یا Elbow نام دارد.

```python
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(7,5))
plt.plot(K, inertia, marker='o', linestyle='--')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()
```

📊 تحلیل:

* معمولاً در Iris، نقطه Elbow در **k=3** دیده می‌شود → که با تعداد کلاس‌های واقعی سازگار است.

---

## 🔹 ۲. روش Silhouette Analysis

شاخص **Silhouette** کیفیت خوشه‌بندی را می‌سنجد.
مقدار آن بین -1 و 1 است:

* نزدیک به **۱** → خوشه‌ها کاملاً جدا از هم.
* نزدیک به **۰** → خوشه‌ها هم‌پوشانی دارند.
* منفی → داده‌ها در خوشه اشتباه قرار گرفته‌اند.

```python
from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in range(2, 11):  # Silhouette نیاز به حداقل 2 خوشه دارد
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# Plot Silhouette Scores
plt.figure(figsize=(7,5))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--', color="green")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis for Optimal k")
plt.show()
```

📊 تحلیل:

* در دیتاست Iris، بیشترین **Silhouette Score** معمولاً در **k=2 یا k=3** اتفاق می‌افتد.
* با توجه به کلاس‌های واقعی (۳ کلاس)، انتخاب **k=3** منطقی‌ترین است.

---

## 🔎 جمع‌بندی مرحله ششم

* روش **Elbow** و **Silhouette** هر دو تأیید می‌کنند که بهترین تعداد خوشه برای Iris حدود **۳** است.
* این نتیجه با برچسب‌های واقعی هم‌خوانی دارد.

---

📌 تا اینجا یک **کتابچه کامل برای K-Means روی Iris** نوشتیم:

1. بارگذاری و تحلیل داده‌ها
2. پیاده‌سازی K-Means
3. Visualization دوبعدی
4. مقایسه با برچسب‌های واقعی
5. مقایسه با مدل‌های نظارت‌شده
6. انتخاب تعداد خوشه بهینه


