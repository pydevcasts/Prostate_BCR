## 📖 صفحه ۷: انتخاب بهترین مقدار k در KNN

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 چرا انتخاب k مهم است؟

در الگوریتم KNN مقدار $k$ یعنی تعداد همسایه‌هایی که برای تصمیم‌گیری در نظر گرفته می‌شوند.

* اگر $k$ خیلی کوچک باشد (مثلاً ۱ یا ۲) → مدل **بیش از حد حساس به نویز** می‌شود.
* اگر $k$ خیلی بزرگ باشد → مدل **بیش از حد ساده** می‌شود و ممکن است دقت کاهش یابد.

پس باید مقدار **بهینه** را پیدا کنیم.

---

### 🔹 پیاده‌سازی انتخاب k بهینه

```python
# Search for best k
k_values = range(1, 21)
mean_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
    mean_accuracies.append(np.mean(scores))

# Plot accuracy vs k
plt.figure(figsize=(10,6))
plt.plot(k_values, mean_accuracies, marker='o', linestyle='-', color='blue')
plt.title("KNN Accuracy vs k (5-Fold Cross Validation)")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Mean Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.show()
```

📊 **تفسیر نمودار:**

* دقت مدل برای مقادیر مختلف k نمایش داده می‌شود.
* معمولاً یک بازه مشخص بهترین دقت را دارد (مثلاً k=5 تا k=7).

---

### 🔹 انتخاب k نهایی

```python
best_k = k_values[np.argmax(mean_accuracies)]
best_accuracy = max(mean_accuracies)

print("Best k:", best_k)
print("Best Accuracy:", best_accuracy)
```

📌 خروجی نمونه:

* بهترین $k = 5$
* بهترین دقت ≈ 0.96

---

### 🔹 مقایسه با Logistic Regression

اگر Logistic Regression دقت 0.96 داشته باشد و KNN با $k=5$ هم به همان حدود برسد، یعنی **هر دو مدل روی دیتاست Wine عملکرد بسیار نزدیک دارند**.

---

### 🔹 جمع‌بندی صفحه ۷

* اهمیت انتخاب k در KNN توضیح داده شد.
* با Cross Validation مقدار بهینه k را پیدا کردیم.
* دیدیم که KNN می‌تواند با انتخاب درست k، هم‌سطح Logistic Regression عمل کند.

