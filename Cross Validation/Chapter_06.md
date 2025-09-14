
## 📖 صفحه ۶: KNN و مقایسه با Logistic Regression

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 معرفی الگوریتم KNN

الگوریتم **K-Nearest Neighbors (KNN)** بر اساس نزدیک‌ترین همسایه‌ها پیش‌بینی می‌کند:

1. برای نمونه جدید، فاصله آن از تمام داده‌های آموزشی محاسبه می‌شود.
2. $k$ نمونه نزدیک‌تر انتخاب می‌شوند.
3. برچسب اکثریت بین آن‌ها به‌عنوان پیش‌بینی انتخاب می‌شود.

فرمول فاصله رایج: **فاصله اقلیدسی**

$$
d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}
$$

---

### 🔹 پیاده‌سازی KNN با Cross Validation

```python
from sklearn.neighbors import KNeighborsClassifier

# Define KNN model with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Perform 5-Fold Cross Validation
scores_knn = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')

print("KNN Accuracy for each fold:", scores_knn)
print("Mean Accuracy:", np.mean(scores_knn))
```

📊 خروجی نمونه:

* Fold 1: 0.94
* Fold 2: 0.96
* Fold 3: 0.92
* Fold 4: 0.93
* Fold 5: 0.95
* **میانگین:** 0.94

---

### 🔹 مقایسه Logistic Regression و KNN

```python
# Compare Logistic Regression and KNN
results = pd.DataFrame({
    "Logistic Regression": scores_logreg,
    "KNN": scores_knn
})

# Plot comparison
plt.figure(figsize=(8,6))
sns.boxplot(data=results, palette="Set2")
plt.title("Comparison of Logistic Regression and KNN (Accuracy in Cross Validation)")
plt.ylabel("Accuracy")
plt.show()
```

📌 **تفسیر احتمالی:**

* Logistic Regression: میانگین حدود 0.96 → پایدار و دقیق
* KNN: میانگین حدود 0.94 → کمی پایین‌تر ولی همچنان خوب

---

### 🔹 نکته مهم

عملکرد KNN به شدت به دو عامل وابسته است:

* مقدار $k$ (تعداد همسایه‌ها)
* نوع فاصله (Euclidean, Manhattan, …)

📍 بنابراین برای بهبود KNN، می‌توانیم در صفحات بعدی **انتخاب بهترین مقدار k با Cross Validation** را بررسی کنیم.

---

### 🔹 جمع‌بندی صفحه ۶

* KNN با Cross Validation اجرا شد.
* نتایج آن با Logistic Regression مقایسه شد.
* Logistic Regression اندکی بهتر بود، اما KNN نیز عملکرد خوبی داشت.
