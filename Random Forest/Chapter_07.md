## 📖 صفحه ۷: آماده‌سازی داده‌ها

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 چرا آماده‌سازی داده‌ها مهم است؟

قبل از آموزش هر مدل یادگیری ماشین، باید داده‌ها را برای استفاده آماده کنیم.
مراحل اصلی شامل:

1. **جداسازی ویژگی‌ها (Features) و برچسب‌ها (Target).**
2. **استانداردسازی (Standardization)** در صورت نیاز (برای الگوریتم‌هایی مثل KNN و SVM بسیار مهم است، برای Random Forest چندان ضروری نیست ولی در مقایسه بین مدل‌ها بهتر است انجام شود).
3. **Train-Test Split یا Cross Validation Setup** برای ارزیابی پایدار مدل.

---

### 🔹 جداسازی Features و Target

```python
# Features (X) and Target (y)
X = df.drop('target', axis=1)
y = df['target']
```

---

### 🔹 استانداردسازی ویژگی‌ها

اگرچه Random Forest نسبت به مقیاس داده حساس نیست، ما به دلیل اینکه قصد مقایسه با مدل‌های دیگر (مثل Logistic Regression, KNN, SVM) را داریم، بهتر است داده‌ها را استانداردسازی کنیم.

```python
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### 🔹 تنظیم Cross Validation

برای ارزیابی پایدار مدل‌ها، از **k-Fold Cross Validation** استفاده می‌کنیم. در این روش داده‌ها به k بخش تقسیم می‌شوند و هر بار یک بخش برای تست و بقیه برای آموزش استفاده می‌شود.

اینجا ما از **۵-Fold Cross Validation** استفاده می‌کنیم:

```python
from sklearn.model_selection import StratifiedKFold

# Define StratifiedKFold (preserves class distribution in folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

### 🔹 چرا StratifiedKFold؟

* چون دیتاست Breast Cancer دو کلاس دارد، مهم است که در هر Fold نسبت Benign و Malignant حفظ شود.
* StratifiedKFold این کار را تضمین می‌کند.

---

### 🔹 نتیجه‌گیری

* داده‌ها به صورت مناسب آماده شدند: ویژگی‌ها استانداردسازی شدند، Target جدا شد.
* Cross Validation با ۵ بخش تنظیم شد تا عملکرد مدل به صورت پایدار و بدون Overfitting سنجیده شود.
* در صفحه بعد، مدل Random Forest را آموزش می‌دهیم و نتایج Cross Validation را تحلیل می‌کنیم.

