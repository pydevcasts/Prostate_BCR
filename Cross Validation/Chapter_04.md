## 📖 صفحه ۴: K-Fold Cross Validation و پیاده‌سازی برای Logistic Regression

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 K-Fold Cross Validation چیست؟

در روش **K-Fold**:

1. داده‌ها به $k$ بخش (Fold) تقسیم می‌شوند.
2. مدل $k$ بار آموزش داده می‌شود: هر بار روی $k-1$ بخش و تست روی بخش باقی‌مانده.
3. در پایان، میانگین همه نتایج گزارش می‌شود.

📌 این روش باعث می‌شود تمام داده‌ها هم برای آموزش و هم برای تست استفاده شوند، فقط در Foldهای مختلف.

---

### 🔹 مزایای K-Fold

* استفاده کامل از داده‌ها
* نتایج پایدارتر نسبت به Train/Test Split ساده
* مناسب برای مقایسه چندین مدل مختلف

---

### 🔹 فرمول دقت نهایی در K-Fold

$$
CV_{score} = \frac{1}{k} \sum_{i=1}^{k} Accuracy_i
$$

---

### 🔹 پیاده‌سازی Logistic Regression با K-Fold

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# Logistic Regression model
logreg = LogisticRegression(max_iter=1000)

# Perform 5-Fold Cross Validation
scores_logreg = cross_val_score(logreg, X_scaled, y, cv=5, scoring='accuracy')

print("Accuracy for each fold:", scores_logreg)
print("Mean Accuracy:", np.mean(scores_logreg))
```

📊 خروجی چیزی شبیه به این خواهد بود:

* Fold 1: 0.97
* Fold 2: 0.94
* Fold 3: 0.95
* Fold 4: 0.96
* Fold 5: 0.97
* **میانگین:** 0.96

---

### 🔹 Visualization دقت Logistic Regression در Cross Validation

```python
plt.figure(figsize=(7,5))
sns.barplot(x=[f"Fold {i+1}" for i in range(len(scores_logreg))], y=scores_logreg, palette="Blues_d")
plt.axhline(np.mean(scores_logreg), color='red', linestyle='--', label=f"Mean Accuracy: {np.mean(scores_logreg):.2f}")
plt.title("Logistic Regression Accuracy in 5-Fold Cross Validation")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

📌 در نمودار، ستون‌ها دقت هر Fold را نشان می‌دهند و خط قرمز میانگین دقت را مشخص می‌کند.

---

### 🔹 نتیجه صفحه ۴

* مفهوم K-Fold را معرفی کردیم.
* Logistic Regression را با Cross Validation پیاده‌سازی کردیم.
* دیدیم که دقت مدل بسیار بالا (حدود ۹۵-۹۷٪) است.
* با نمودار، تغییرات دقت در Foldهای مختلف را نمایش دادیم.
