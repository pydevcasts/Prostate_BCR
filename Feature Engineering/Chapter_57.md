
# ⚙️ فصل ۱۲ — ساخت ویژگی‌های جدید

## 📄 صفحه ۳ از ۵ — تبدیل ویژگی‌ها (Feature Transformations)

✍️ *نویسنده: سیامک عباس‌نژاد*
🌐 *[https://github.com/pydevcasts](https://github.com/pydevcasts)*

---

## 🎯 هدف این صفحه

در پایان این بخش یاد می‌گیری:
✅ چرا تبدیل ویژگی مهم است
✅ تبدیل‌های لگاریتمی، ریشه‌ای، نمایی
✅ نرمال‌سازی و استانداردسازی
✅ تبدیل‌های Box-Cox و Yeo-Johnson
✅ ویژگی‌های چندجمله‌ای (Polynomial Features)
✅ مثال‌های عددی
✅ کدهای پایتون

---

# 🧠 ۱. تبدیل ویژگی (Feature Transformation) یعنی چه؟

«تبدیل ویژگی» یعنی **تغییر شکل دادن داده‌ها** برای اینکه:

* الگوها ساده‌تر شوند
* مدل راحت‌تر یاد بگیرد
* توزیع داده متعادل شود
* داده‌های پرت اثرش کمتر شود
* روابط غیرخطی خطی شوند

مثال ساده:

اگر درآمد کاربران اینطوری باشد:

```
3, 4, 10, 50, 200, 3000
```

کاملاً پخش نامتعادل است → مدل گیج می‌شود!
اما اگر Log بگیری:

```
log → 1.1 , 1.3 , 2.3 , 3.9 , 5.3 , 8.0
```

ناگهان *تعادل ایجاد می‌شود* و مدل بهتر کار می‌کند.

---

# 📊 ۲. تبدیل لگاریتمی (Log Transformation)

🔹 بهترین روش برای داده‌های **Long Tail**
🔹 مناسب داده‌هایی با اختلاف زیاد
🔹 کمک می‌کند رابطه‌ها «خطی‌تر» شوند

مثال عددی:

| درآمد | log(درآمد) |
| ----- | ---------- |
| 100   | 4.6        |
| 5000  | 8.5        |

اختلاف شدید 100 → 5000 تبدیل شد به اختلاف کوچک 4.6 → 8.5
این یعنی **ثبات بیشتر در یادگیری مدل**.

---

# 🔢 ۳. تبدیل ریشه‌ای (Square Root)

به درد داده‌هایی می‌خورد که کمی skewed هستند.

```
sqrt(100) = 10
sqrt(400) = 20
```

---

# 📈 ۴. تبدیل نمایی (Exponential)

کمتر استفاده می‌شود، اما برای بزرگ‌کردن تفاوت‌های کوچک عالی است.

```
exp(2) = 7.39
exp(5) = 148.4
```

---

# 🔄 ۵. نرمال‌سازی (Normalization)

داده را در بازه‌ی `[0, 1]` قرار می‌دهد.

```
x_norm = (x - min) / (max - min)
```

مناسب برای الگوریتم‌هایی مثل:

* KNN
* Neural Networks

---

# 📏 ۶. استانداردسازی (Standardization)

داده را تبدیل به میانگین صفر و واریانس یک می‌کند:

```
x_std = (x - mean) / std
```

مناسب برای:

* Logistic Regression
* SVM
* Linear Models

---

# 🔁 ۷. روش Box-Cox Transformation

یک تبدیل پیشرفته برای نرمال‌کردن داده‌های مثبت.

```
اگر λ = 0 → تبدیل لگاریتمی  
اگر λ ≠ 0 → (x^λ - 1) / λ
```

در sklearn:

```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='box-cox')
```

---

# 🌀 ۸. روش Yeo-Johnson

نسخهٔ بهتر Box-Cox چون داده‌های **منفی** را هم پشتیبانی می‌کند.

در sklearn:

```python
pt = PowerTransformer(method='yeo-johnson')
```

---

# 🧮 ۹. ویژگی‌های چندجمله‌ای (Polynomial Features)

مهم‌ترین قسمت این صفحه!

از ویژگی‌های ساده، ویژگی‌های جدید غیرخطی می‌سازد:

```
x
x^2
x^3
x1 * x2
x1^2 * x2
```

در sklearn:

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
new_features = poly.fit_transform(data)
```

کاربردها:

* رگرسیون غیرخطی
* الگوهای پیچیده
* تشخیص خمیدگی داده

---

# 💻 ۱۰. کد پایتون — اجرای چند تبدیل روی یک دیتافریم

⚠️ شروع هر خط فارسی تا چپ‌چین نشود.

```python

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures, PowerTransformer

data = pd.DataFrame({
    "income": [3, 5, 12, 60, 400, 3000],
    "age": [20, 25, 30, 35, 40, 45]
})

# لگاریتم
data["income_log"] = np.log1p(data["income"])

# ریشه
data["income_sqrt"] = np.sqrt(data["income"])

# استانداردسازی
sc = StandardScaler()
data["age_std"] = sc.fit_transform(data[["age"]])

# Box-Cox (فقط برای داده مثبت)
pt = PowerTransformer(method="box-cox")
data["income_boxcox"] = pt.fit_transform(data[["income"]])

# ویژگی چندجمله‌ای
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_out = poly.fit_transform(data[["age"]])
data["age_squared"] = poly_out[:, 1]

print(data)
```

---

# 🎨 ۱۱. تصویر پیشنهادی

> نمودار توزیع داده قبل و بعد از Log Transform
> (قبل → skewed شدید، بعد → متعادل و زیبا)

---

# 🧠 ۱۲. تمرین

۱) یک ویژگی با توزیع نامتعادل پیدا کن
۲) روی آن سه تبدیل زیر انجام بده:

* Log
* Standardization
* Box-Cox
  ۳) توزیع قبل و بعد را بکش
  ۴) دقت مدل را قبل/بعد مقایسه کن

---

# ❓ آزمون چهارگزینه‌ای صفحه

**کدام تبدیل برای داده‌های مثبت و بسیار skewed بهترین انتخاب است؟**

A) Standardization

B) MinMax Scaling

C) Log Transform ✅

D) One-Hot Encoding
