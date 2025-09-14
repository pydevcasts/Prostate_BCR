## 📖 صفحه ۸: SVM و مقایسه سه مدل در Cross Validation

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 معرفی SVM

الگوریتم **Support Vector Machine (SVM)** یکی از قدرتمندترین الگوریتم‌های طبقه‌بندی است.
ایده اصلی: پیدا کردن یک **مرز تصمیم‌گیری (Hyperplane)** که کلاس‌ها را از هم جدا کند.

فرمول ساده‌شده برای خط مرزی:

$$
w \cdot x + b = 0
$$

که در آن:

* $w$: بردار وزن‌ها
* $b$: بایاس
* $x$: داده ورودی

SVM سعی می‌کند **فاصله (Margin)** بین کلاس‌ها را حداکثر کند.

---

### 🔹 پیاده‌سازی SVM با Cross Validation

```python
from sklearn.svm import SVC

# Define SVM model (linear kernel for start)
svm = SVC(kernel='linear')

# Perform 5-Fold Cross Validation
scores_svm = cross_val_score(svm, X_scaled, y, cv=5, scoring='accuracy')

print("SVM Accuracy for each fold:", scores_svm)
print("Mean Accuracy:", np.mean(scores_svm))
```

📌 خروجی نمونه:

* Fold 1: 0.97
* Fold 2: 0.96
* Fold 3: 0.98
* Fold 4: 0.95
* Fold 5: 0.97
* **میانگین:** 0.97

---

### 🔹 مقایسه سه مدل (Logistic Regression, KNN, SVM)

```python
# Combine results
results_all = pd.DataFrame({
    "Logistic Regression": scores_logreg,
    "KNN": scores_knn,
    "SVM": scores_svm
})

# Boxplot comparison
plt.figure(figsize=(9,6))
sns.boxplot(data=results_all, palette="Set3")
plt.title("Comparison of Logistic Regression, KNN, and SVM (Accuracy in Cross Validation)")
plt.ylabel("Accuracy")
plt.show()
```

📊 **تفسیر احتمالی نمودار:**

* Logistic Regression: میانگین ≈ 0.96
* KNN (با k بهینه): میانگین ≈ 0.96
* SVM: میانگین ≈ 0.97 → بهترین عملکرد

---

### 🔹 نکته مهم

* SVM معمولاً در دیتاست‌هایی با ویژگی‌های زیاد (مثل Wine) عملکرد عالی دارد.
* KNN نیاز به انتخاب k دارد.
* Logistic Regression ساده‌تر و سریع‌تر است ولی ممکن است در دیتاست‌های پیچیده کمی ضعیف‌تر باشد.

---

### 🔹 جمع‌بندی صفحه ۸

* SVM معرفی و با Cross Validation اجرا شد.
* نتایج سه مدل در یک نمودار مقایسه شدند.
* نتیجه: **SVM کمی بهتر از Logistic Regression و KNN عمل کرد.**
