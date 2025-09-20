## 📖 صفحه ۱۱: مقایسه Random Forest با مدل‌های دیگر

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 چرا مقایسه مهم است؟

هر الگوریتم یادگیری ماشین مزایا و محدودیت‌های خودش را دارد. برای اطمینان از انتخاب بهترین مدل، باید چند الگوریتم مختلف را با شرایط یکسان (همان دیتاست و همان Cross Validation) مقایسه کنیم.

در این بخش، چهار مدل زیر را بررسی می‌کنیم:

1. **Random Forest** 🌲
2. **Logistic Regression** 📈
3. **K-Nearest Neighbors (KNN)** 👥
4. **Support Vector Machine (SVM)** ⚖️

---

### 🔹 پیاده‌سازی مقایسه با Cross Validation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", random_state=42,probability=True)
}

# Dictionary to store results
cv_results = {}

# Evaluate each model
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")
    cv_results[name] = scores
    print(f"{name} -> Mean Accuracy: {scores.mean():.4f} | Std: {scores.std():.4f}")
```

---

### 🔹 نمونه خروجی (مثال)

```
Random Forest -> Mean Accuracy: 0.9561 | Std: 0.0123
Logistic Regression -> Mean Accuracy: 0.9737 | Std: 0.0166
KNN -> Mean Accuracy: 0.9666 | Std: 0.0140
SVM -> Mean Accuracy: 0.9754 | Std: 0.0195
```

---

### 🔹 تفسیر نتایج

* **Random Forest (95%)** → قابل قبول، ولی نسبت به مدل‌های دیگر کمی ضعیف‌تر.
* **SVM (97.5%)** → بهترین عملکرد، پایدار و مقاوم.
 → عملکرد نزدیک به Random Forest، ولی کمی پایین‌تر.
* **Logistic Regression (97%)** → ساده و سریع، عملکرد خوب ولی نه به اندازه SVM.
* **KNN (96%)** → عملکرد بهتری نسبت به جنگل تصادفی داشت

---

### 🔹 مقدمه برای صفحه بعد

در صفحه بعد، نتایج Cross Validation را به صورت تصویری (Barplot + Boxplot) مقایسه می‌کنیم تا تحلیل دیداری دقیق‌تری داشته باشیم.

