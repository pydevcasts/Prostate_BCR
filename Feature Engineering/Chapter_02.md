# 📘 فصل دوم: بارگذاری و بررسی داده‌ها

🔹 در این فصل می‌خواهیم دیتاست **سرطان سینه (Breast Cancer Dataset)** را از کتابخانه‌ی scikit-learn بارگذاری کنیم و یک بررسی اولیه روی داده‌ها انجام بدهیم. این مرحله اولین گام عملی در مسیر مهندسی ویژگی‌ها است.

---

## 📥 بارگذاری داده

ابتدا با استفاده از کتابخانه‌ی **scikit-learn** دیتاست را بارگذاری می‌کنیم و آن را به یک DataFrame از نوع **pandas** تبدیل می‌کنیم:

```python
# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Load Breast Cancer dataset
data = load_breast_cancer()

# Convert to pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # Add target column
df.head()
```

📌 در اینجا ستون `target` نشان‌دهنده‌ی کلاس خروجی است:

* **0 = Malignant (بدخیم)**
* **1 = Benign (خوش‌خیم)**

---

## 📊 بررسی اولیه داده‌ها

### ۱. ابعاد داده‌ها

```python
df.shape
```

🔹 خروجی: `(569, 31)`
یعنی ۵۶۹ نمونه و ۳۰ ویژگی به همراه ستون هدف (`target`).

---

### ۲. بررسی داده‌های گمشده

```python
df.isnull().sum().head()
```

🔹 خوشبختانه این دیتاست **فاقد داده‌ی گمشده** است.

---

### ۳. توصیف آماری داده‌ها

```python
df.describe().T.head()
```

🔹 این دستور میانگین، انحراف معیار، حداقل و حداکثر هر ویژگی را نمایش می‌دهد.
این آمار به ما کمک می‌کند که بفهمیم کدام ویژگی‌ها مقادیر بزرگ‌تر یا کوچک‌تری دارند.

---

## 📌 بررسی تعادل کلاس‌ها

خیلی مهم است بدانیم چند نمونه بدخیم و چند نمونه خوش‌خیم داریم.

```python
sns.countplot(x='target', data=df, palette='Set2')
plt.xticks([0, 1], ['Malignant', 'Benign'])
plt.title("Distribution of Classes (Target)")
plt.show()
```

📊 نمودار نشان می‌دهد که:

* حدود **۲۱۲ نمونه بدخیم (Malignant)**
* حدود **۳۵۷ نمونه خوش‌خیم (Benign)**

این تعادل نسبتاً خوب است و مشکل جدی در نامتوازنی کلاس‌ها نداریم.

