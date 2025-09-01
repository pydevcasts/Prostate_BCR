
## 📖 صفحه ۶: تحلیل داده‌ها (EDA) – بخش اول

✍️ نویسنده: سیامک عباس‌نژاد

قبل از ساخت مدل رگرسیون لجستیک، لازم است داده‌ها را به‌خوبی بررسی کنیم. این فرآیند را در علم داده با نام **تحلیل اکتشافی داده‌ها (Exploratory Data Analysis = EDA)** می‌شناسیم. هدف این مرحله، شناخت بهتر ساختار داده، بررسی مقادیر گمشده، درک توزیع ویژگی‌ها و کشف روابط احتمالی بین متغیرهاست.

---

### 🔹 بارگذاری و بررسی اولیه داده‌ها

در این پروژه از پایتون و کتابخانه‌های **Pandas**, **NumPy**, **Matplotlib** و **Seaborn** استفاده می‌کنیم. در کد زیر داده‌ها بارگذاری شده و بررسی اولیه انجام می‌شود:

```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("diabetes.csv")

# Display first 5 rows
print(data.head())

# Dataset info
print(data.info())

# Summary statistics
print(data.describe())
```

---

### 🔹 آمار توصیفی اولیه

* ویژگی‌هایی مثل **Pregnancies** و **Age** مقادیر صحیح (Integer) دارند.
* ویژگی‌هایی مثل **Glucose**, **BloodPressure**, **BMI** پیوسته (Continuous) هستند.
* در جدول آماری (`describe()`) مشاهده می‌شود که بعضی ویژگی‌ها مانند **Insulin** و **SkinThickness** شامل مقادیر صفر غیرواقعی هستند که باید در مراحل پیش‌پردازش اصلاح شوند.

---

### 🔹 توزیع داده‌ها

یکی از اولین کارهایی که باید انجام دهیم، بررسی توزیع متغیرهاست. برای این کار می‌توانیم از هیستوگرام (Histogram) استفاده کنیم:

```python
# Plot histograms for each feature
data.hist(figsize=(12,10), bins=20)
plt.tight_layout()
plt.show()
```

📊 این نمودار نشان می‌دهد که برخی ویژگی‌ها مانند **Glucose** توزیع نسبتاً نرمال دارند، اما ویژگی‌هایی مثل **Insulin** دارای مقادیر بسیار زیادی صفر هستند که نیاز به پردازش دارند.

---

### 🔹 بررسی تعادل برچسب‌ها (Outcome)

خیلی مهم است بدانیم که آیا تعداد بیماران دیابتی و غیردیابتی در دیتاست متعادل است یا خیر.

```python
# Count plot of target variable
sns.countplot(x="Outcome", data=data)
plt.title("Distribution of Diabetes Outcome")
plt.show()
```

📌 نتیجه: در این دیتاست حدود **۵۰۰ نفر غیرمبتلا (۰)** و حدود **۲۶۸ نفر مبتلا (۱)** هستند. یعنی داده‌ها نسبتاً نامتوازن هستند، ولی هنوز قابل استفاده‌اند.

---

📍 در صفحه بعد (صفحه 7)، تحلیل داده‌ها را ادامه می‌دهیم و با استفاده از **Correlation Matrix** و نمودارهای جفتی (Pair Plots) روابط میان ویژگی‌ها را بررسی خواهیم کرد.

---

