## 📖 صفحه ۸: آموزش مدل Random Forest با Cross Validation

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 آموزش مدل Random Forest

اکنون که داده‌ها آماده شدند، مدل **Random Forest** را با استفاده از **۵-Fold Cross Validation** آموزش می‌دهیم. هدف ما بررسی عملکرد مدل در هر Fold و سپس میانگین نتایج است.

---

### 🔹 کد آموزش و ارزیابی

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Define Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross Validation (Accuracy)
scores = cross_val_score(rf_model, X_scaled, y, cv=cv, scoring='accuracy')

# Print accuracy for each fold
print("Accuracy for each fold:", scores)
print("Mean Accuracy:", np.mean(scores))
print("Standard Deviation:", np.std(scores))
```

---

### 🔹 توضیح کد

* **n\_estimators=100**: تعداد درخت‌ها در جنگل را مشخص می‌کند (اینجا ۱۰۰).
* **cross\_val\_score**: مدل را روی داده‌ها آموزش می‌دهد و با استفاده از Cross Validation عملکرد را می‌سنجد.
* **np.mean(scores)**: میانگین دقت‌ها.
* **np.std(scores)**: میزان تغییرپذیری عملکرد مدل در Foldهای مختلف.

---

### 🔹 نمونه خروجی (مثال)

```
Accuracy for each fold: [0.96 0.94 0.95 0.97 0.96]
Mean Accuracy: 0.96
Standard Deviation: 0.01
```

🔸 نتیجه نشان می‌دهد که مدل Random Forest روی دیتاست **Breast Cancer** به دقت حدود **۹۶٪** رسیده است.
🔸 انحراف معیار بسیار پایین است (۰.۰۱)، که یعنی مدل پایدار و قابل اعتماد است.

---

### 🔹 تفسیر نتایج

* **دقت بالا (۹۶٪)**: نشان می‌دهد Random Forest توانسته به خوبی الگوهای داده را تشخیص دهد.
* **انحراف معیار پایین**: عملکرد مدل روی Foldهای مختلف تقریباً یکسان است → مدل Generalize خوبی دارد.

---

### 🔹 مقدمه برای صفحه بعد

در صفحه بعد، نتایج Cross Validation را به صورت تصویری نمایش می‌دهیم (Barplot و Boxplot) تا تحلیل دیداری بهتری داشته باشیم.
