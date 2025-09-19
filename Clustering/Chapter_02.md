# 📖 فصل دوم: الگوریتم K-Means و پیاده‌سازی

الگوریتم **K-Means** یکی از ساده‌ترین و در عین حال پرکاربردترین روش‌های خوشه‌بندی است.

---

## 🔹 مراحل الگوریتم K-Means

1. انتخاب تعداد خوشه‌ها (K).
2. انتخاب تصادفی K مرکز اولیه.
3. محاسبه فاصله‌ی هر نقطه از مراکز خوشه (معمولاً فاصله اقلیدسی).
4. انتساب هر نقطه به نزدیک‌ترین خوشه.
5. به‌روزرسانی مراکز خوشه با میانگین نقاط هر گروه.
6. تکرار مراحل ۳ تا ۵ تا زمانی که مراکز خوشه تغییر نکنند یا تعداد تکرارها تمام شود.

---

## 📐 فرمول فاصله اقلیدسی

$$
d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}
$$

---

## 📊 پیاده‌سازی K-Means روی دیتاست Iris

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Features only
X = df.drop("target", axis=1)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans with K=3
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

df[['target', 'cluster']].head()
```

---

## 📊 مقایسه خوشه‌ها با برچسب‌های واقعی

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,1], hue=df['cluster'], palette="viridis", style=df['target'])
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("KMeans Clustering vs Actual Classes")
plt.show()
```

📌 مشاهده:

* خوشه‌ی مربوط به **Setosa** کاملاً جداست.
* اما دو کلاس **Versicolor** و **Virginica** با هم تداخل دارند.

---

## 📊 ارزیابی خوشه‌بندی

برای ارزیابی، چون برچسب واقعی داریم، می‌توانیم از **Accuracy** یا **Adjusted Rand Index (ARI)** استفاده کنیم.

```python
from sklearn.metrics import adjusted_rand_score

score = adjusted_rand_score(df['target'], df['cluster'])
print("ARI Score:", score)
```

📌 مقدار ARI نزدیک به **۰.۷** یا بالاتر می‌تواند نشان دهد که خوشه‌بندی نسبتاً خوب عمل کرده است.

