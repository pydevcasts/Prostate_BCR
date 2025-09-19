## 📖 صفحه ۹: آموزش مدل رگرسیون لجستیک و ارزیابی اولیه

✍️ نویسنده: سیامک عباس‌نژاد

بعد از پیش‌پردازش داده‌ها، حالا نوبت به بخش اصلی می‌رسد: **آموزش مدل رگرسیون لجستیک**. این مدل یکی از ساده‌ترین و درعین‌حال پرکاربردترین الگوریتم‌ها در مسائل دسته‌بندی (Classification) است.

---

### 🔹 آموزش مدل رگرسیون لجستیک

در این مرحله مدل را ساخته و با داده‌های آموزش (Training set) آن را یاد می‌دهیم:

```python
from sklearn.linear_model import LogisticRegression

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)
```

---

### 🔹 پیش‌بینی با مدل

پس از آموزش، می‌توانیم پیش‌بینی‌های مدل را روی داده‌های تست (Test set) انجام دهیم:

```python
# Predictions
y_pred = model.predict(X_test)
```

---

### 🔹 ارزیابی اولیه مدل

برای بررسی کیفیت مدل از معیارهایی مانند **Accuracy (دقت)**، **Confusion Matrix (ماتریس آشفتگی)** و **Classification Report (گزارش دسته‌بندی)** استفاده می‌کنیم.

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))
```

---

### 🔹 تحلیل نتایج اولیه

* **Accuracy** معمولاً حدود ۷۰٪ تا ۷۵٪ خواهد بود.
* ماتریس آشفتگی نشان می‌دهد که چه تعداد نمونه‌ها درست و غلط پیش‌بینی شده‌اند.
* گزارش دسته‌بندی شامل معیارهای مهم **Precision، Recall و F1-score** است که نشان‌دهنده‌ی توانایی مدل در تشخیص درست بیماران مبتلا و غیرمبتلا به دیابت است.

---

### 🔹 اهمیت متریک‌ها

* **Precision**: چه تعداد از پیش‌بینی‌های "دیابتی" واقعاً دیابتی بودند.
* **Recall**: چه تعداد از بیماران واقعی دیابتی به‌درستی شناسایی شدند.
* **F1-score**: میانگین هماهنگ بین Precision و Recall.

---

📍 در صفحه بعد (**صفحه ۹**) به **بهبود مدل** می‌پردازیم. در این بخش از **Feature Selection**، **نرمال‌سازی پارامترها** و مقایسه‌ی عملکرد مدل استفاده خواهیم کرد تا ببینیم آیا دقت و کیفیت پیش‌بینی افزایش می‌یابد یا خیر.

