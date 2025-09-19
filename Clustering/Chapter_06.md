
# 📖 فصل ششم: مقایسه الگوریتم‌های خوشه‌بندی

در این فصل می‌خواهیم سه الگوریتمی که بررسی کردیم (K-Means، Hierarchical و DBSCAN) را با هم مقایسه کنیم.

---

## 🔹 معیارهای ارزیابی خوشه‌بندی

چون خوشه‌بندی یک روش **بدون نظارت** است، معمولاً داده‌های واقعی برچسب ندارند.
اما در دیتاست **Iris** چون برچسب داریم، می‌توانیم از روش‌های زیر استفاده کنیم:

* **Adjusted Rand Index (ARI)**: مقایسه خوشه‌ها با برچسب‌های واقعی
* **Silhouette Score**: میزان جداسازی خوشه‌ها صرف‌نظر از برچسب واقعی
* **Homogeneity, Completeness, V-Measure**: شباهت خوشه‌ها به دسته‌های واقعی

---

## 📊 پیاده‌سازی مقایسه

```python
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# Define models
models = {
    "KMeans": KMeans(n_clusters=3, random_state=42),
    "Hierarchical": AgglomerativeClustering(n_clusters=3, linkage="ward"),
    "DBSCAN": DBSCAN(eps=0.8, min_samples=5)
}

results = []

for name, model in models.items():
    labels = model.fit_predict(X_scaled)
    ari = adjusted_rand_score(df['target'], labels)
    sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1
    results.append([name, ari, sil])

# Create results DataFrame
results_df = pd.DataFrame(results, columns=["Model", "ARI", "Silhouette"])
print(results_df)
```

---

## 📊 ترسیم نمودار مقایسه

```python
# Barplot for comparison
plt.figure(figsize=(10,6))
sns.barplot(x="Model", y="ARI", data=results_df, palette="Set2")
plt.title("Adjusted Rand Index Comparison")
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x="Model", y="Silhouette", data=results_df, palette="Set1")
plt.title("Silhouette Score Comparison")
plt.show()
```

📌 مشاهده:

* **KMeans** و **Hierarchical** عملکرد مشابهی دارند و در جداسازی **Setosa** عالی هستند.
* **DBSCAN** توانسته نویزها را شناسایی کند اما در تفکیک Versicolor و Virginica ضعیف‌تر است.

---

## ✨ جمع‌بندی نهایی

* اگر داده‌ها **ساختار ساده و کروی** دارند ➝ KMeans بهترین انتخاب است.
* اگر داده‌ها **کوچک و قابل تجسم** هستند ➝ Hierarchical مناسب است (به‌ویژه برای دندروگرام).
* اگر داده‌ها شامل **شکل‌های پیچیده و نویز** باشند ➝ DBSCAN انتخاب بهتری است.

