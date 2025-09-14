## 📖 صفحه ۲: معرفی دیتاست Breast Cancer

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 معرفی دیتاست

دیتاست **Breast Cancer Wisconsin (Diagnostic)** یکی از دیتاست‌های پرکاربرد در یادگیری ماشین است که برای شناسایی و تشخیص سرطان سینه استفاده می‌شود. این دیتاست در کتابخانه‌ی **scikit-learn** موجود است و شامل اطلاعات واقعی از بیماران می‌باشد.

مشخصات اصلی دیتاست:

* **تعداد نمونه‌ها:** 569
* **تعداد ویژگی‌ها:** 30 ویژگی عددی
* **تعداد کلاس‌ها:** 2 کلاس

  * Malignant (سرطانی) → برچسب **1**
  * Benign (خوش‌خیم) → برچسب **0**

هر ویژگی در این دیتاست مربوط به اندازه‌گیری‌های مختلف سلول‌هاست، مثل:

* شعاع (Radius)
* بافت (Texture)
* محیط (Perimeter)
* مساحت (Area)
* همواری (Smoothness)

---

### 🔹 بارگذاری دیتاست با پایتون

```python
# Import libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load Breast Cancer dataset
data = load_breast_cancer()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Show first 5 rows
print(df.head())
```

🔹 خروجی شامل چند ردیف اول دیتاست به شکل زیر خواهد بود:

| mean radius | mean texture | mean perimeter | … | mean fractal dimension | target |
| ----------- | ------------ | -------------- | - | ---------------------- | ------ |
| 17.99       | 10.38        | 122.80         | … | 0.11890                | 0      |
| 20.57       | 17.77        | 132.90         | … | 0.08902                | 0      |
| 19.69       | 21.25        | 130.00         | … | 0.08758                | 0      |

---

### 🔹 بررسی اولیه توزیع کلاس‌ها

برای اینکه مطمئن شویم کلاس‌ها متعادل هستند یا نه، توزیع برچسب‌ها را بررسی می‌کنیم:

```python
# Count target values
sns.countplot(x='target', data=df, palette='coolwarm')
plt.title("Distribution of Target Classes (0 = Benign, 1 = Malignant)")
plt.show()
```

📊 نتیجه:

* تقریباً ۲۱۲ نمونه Malignant (سرطانی)
* تقریباً ۳۵۷ نمونه Benign (خوش‌خیم)

بنابراین، دیتاست نسبتاً متوازن است و می‌تواند برای آموزش مدل مناسب باشد.

---

### 🔹 اهمیت دیتاست در یادگیری ماشین

این دیتاست به دلیل داشتن ویژگی‌های عددی دقیق و دو کلاس مشخص، یکی از بهترین دیتاست‌ها برای یادگیری الگوریتم‌های طبقه‌بندی مانند **Random Forest، Logistic Regression، KNN، و SVM** است.
در ادامه، ما از آن برای آموزش و تحلیل الگوریتم Random Forest استفاده خواهیم کرد.

