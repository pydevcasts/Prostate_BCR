
## 📖 صفحه ۴: پیاده‌سازی پروژه با پایتون (SVM برای تشخیص سرطان سینه)

✍️ نویسنده: سیامک عباس‌نژاد

در این بخش، پروژه‌ی تشخیص سرطان سینه با استفاده از الگوریتم **ماشین بردار پشتیبان (SVM)** را پیاده‌سازی می‌کنیم. برای این کار از زبان **پایتون** و کتابخانه‌های محبوب دیتا ساینس شامل `numpy`, `pandas`, `matplotlib` و `scikit-learn` استفاده می‌کنیم.

کلیه‌ی توضیحات کدها (کامنت‌ها) به زبان انگلیسی نوشته شده‌اند تا برای خوانندگان فنی در سطح بین‌المللی نیز قابل فهم باشند، اما متن کتاب فارسی و راست‌چین باقی می‌ماند.

---

### 🔹 کدنویسی پروژه

```python
# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Breast Cancer dataset from sklearn
data = datasets.load_breast_cancer()
X = data.data      # Features
y = data.target    # Labels (0 = Malignant, 1 = Benign)

# Split dataset into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM model with linear kernel
linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)

# Predict on test data
y_pred_linear = linear_svm.predict(X_test)

# Evaluate performance
print("Accuracy with Linear Kernel:", accuracy_score(y_test, y_pred_linear))
print("Confusion Matrix (Linear Kernel):\n", confusion_matrix(y_test, y_pred_linear))
print("Classification Report (Linear Kernel):\n", classification_report(y_test, y_pred_linear))

# Train SVM model with RBF kernel
rbf_svm = SVC(kernel='rbf')
rbf_svm.fit(X_train, y_train)

# Predict on test data
y_pred_rbf = rbf_svm.predict(X_test)

# Evaluate performance
print("\nAccuracy with RBF Kernel:", accuracy_score(y_test, y_pred_rbf))
print("Confusion Matrix (RBF Kernel):\n", confusion_matrix(y_test, y_pred_rbf))
print("Classification Report (RBF Kernel):\n", classification_report(y_test, y_pred_rbf))
```

---

### 🔹 توضیح کدها

۱. ابتدا کتابخانه‌های لازم را بارگذاری کردیم.

۲. دیتاست سرطان سینه را از `sklearn.datasets` فراخوانی کردیم.

۳. داده‌ها به دو بخش **آموزش** و **آزمون** تقسیم شدند (۸۰٪ آموزش و ۲۰٪ تست).

۴. داده‌ها نرمال‌سازی (Standardization) شدند؛ این مرحله برای عملکرد بهتر SVM حیاتی است.

۵. یک مدل **SVM با کرنل خطی** ساخته و آموزش داده شد، سپس روی داده‌های تست ارزیابی شد.

۶. همان کار را برای **کرنل RBF** نیز انجام دادیم تا نتایج مقایسه شوند.

۷. خروجی شامل **دقت (Accuracy)**، **ماتریس درهم‌ریختگی (Confusion Matrix)** و **گزارش طبقه‌بندی (Classification Report)** است.

---

📍 در صفحه بعدی (صفحه ۵)، نتایج این پیاده‌سازی را تحلیل کرده و جمع‌بندی نهایی کتاب را ارائه می‌کنیم.

---
