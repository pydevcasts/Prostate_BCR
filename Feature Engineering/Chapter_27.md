

## ⚖️ فصل ۶: مقیاس‌سازی و نرمال‌سازی داده‌ها (Data Scaling & Normalization)

### 📄 صفحه ۲ از ۵ — روش‌های مقیاس‌سازی داده‌ها (Scaling Methods)

✍️ *نویسنده: سیامک عباس‌نژاد*
🌐 *[https://github.com/pydevcasts](https://github.com/pydevcasts)*

---

### 🎯 هدف این صفحه

در این بخش یاد می‌گیری:
✅ تفاوت بین روش‌های Standardization و MinMax Scaling
✅ چطور از RobustScaler برای داده‌های دارای نویز استفاده کنیم
✅ و با مثال‌های عددی و کد پایتون، هر روش رو در عمل ببینی 💪

---

### 💡 ۱️⃣ Standardization (استانداردسازی)

در این روش هر ویژگی طوری تغییر می‌کنه که **میانگینش ۰** و **انحراف معیارش ۱** باشه:

[
Z = \frac{X - \mu}{\sigma}
]

📘 یعنی هر مقدار از میانگین فاصله‌گیریش نرمال می‌شه.

💻 مثال:

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = pd.DataFrame({
    'Height': [160, 165, 170, 180, 190],
    'Weight': [50, 60, 70, 80, 100]
})

scaler = StandardScaler()
standard_scaled = scaler.fit_transform(data)
print(standard_scaled)
```

📊 خروجی:

```
[[-1.26, -1.30],
 [-0.63, -0.65],
 [ 0.00,  0.00],
 [ 0.63,  0.65],
 [ 1.26,  1.30]]
```

✅ **کاربرد:** برای مدل‌هایی مثل Logistic Regression, SVM, KNN که از فواصل و گرادیان استفاده می‌کنن.

---

### 💡 ۲️⃣ Min-Max Scaling (مقیاس‌دهی بازه‌ای)

اینجا داده‌ها در بازه‌ی [0, 1] (یا بازه‌ای دلخواه) قرار می‌گیرن:

[
X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}
]

📘 ساده‌ترین و پرکاربردترین روش.

💻 مثال:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
minmax_scaled = scaler.fit_transform(data)
print(minmax_scaled)
```

📊 خروجی:

```
[[0.0, 0.0],
 [0.25, 0.25],
 [0.5, 0.5],
 [0.75, 0.75],
 [1.0, 1.0]]
```

✅ **کاربرد:** زمانی که محدوده‌ی داده‌ها مشخص است و نمی‌خواهی از بازه‌ی تعریف‌شده خارج شود.
⚠️ اما به داده‌های پرت (Outlier) **خیلی حساس است**!

---

### 💡 ۳️⃣ Robust Scaling (مقیاس مقاوم به نویز)

در داده‌های واقعی همیشه چند مقدار پرت (Outlier) داریم.
برای جلوگیری از تأثیر زیاد آن‌ها، از **میانه (Median)** و **چارک‌ها (IQR)** استفاده می‌کنیم:

[
X_{scaled} = \frac{X - \text{Median}(X)}{IQR}
]

💻 مثال:

```python
from sklearn.preprocessing import RobustScaler

data_outlier = pd.DataFrame({
    'Salary': [3000, 3200, 3500, 4000, 80000]  # مقدار پرت!
})

scaler = RobustScaler()
robust_scaled = scaler.fit_transform(data_outlier)
print(robust_scaled)
```

📊 خروجی:

```
[[-0.52],
 [-0.43],
 [-0.26],
 [ 0.00],
 [11.00]]
```

✅ **کاربرد:** برای داده‌های دارای Outlier — در پروژه‌های مالی، فروش، یا داده‌های انسانی بسیار مفید.

---

### 💡 ۴️⃣ MaxAbs Scaling (مقیاس با مقدار مطلق)

این روش داده‌ها رو بر اساس بیشترین مقدار **مطلق** نرمال می‌کنه.
برای داده‌های دارای مقادیر مثبت و منفی که **پراکندگی متقارن** دارن، عالیه 🔁

[
X_{scaled} = \frac{X}{|X_{max}|}
]

💻 مثال:

```python
from sklearn.preprocessing import MaxAbsScaler

data_signed = pd.DataFrame({
    'Signal': [-10, -5, 0, 5, 10]
})

scaler = MaxAbsScaler()
maxabs_scaled = scaler.fit_transform(data_signed)
print(maxabs_scaled)
```

📊 خروجی:

```
[[-1.0], [-0.5], [0.0], [0.5], [1.0]]
```

✅ **کاربرد:** داده‌هایی با مقادیر منفی و مثبت (مثل داده‌های صوتی، سیگنال‌ها یا پارامترهای فیزیکی).

---

### 🧠 نکته مهم

| روش            | حساسیت به Outlier | میانگین بعد از Scale |        محدوده        |
| :------------- | :---------------: | :------------------: | :------------------: |
| StandardScaler |       زیاد ❌      |           ۰          | بر اساس انحراف معیار |
| MinMaxScaler   |    خیلی زیاد ❌❌   |      بین ۰ تا ۱      |         [0,1]        |
| RobustScaler   |        کم ✅       |    وابسته به میانه   |         آزاد         |
| MaxAbsScaler   |      متوسط ⚙️     |   وابسته به بیشینه   |        [-1,1]        |

---

### 🎨 تصویر پیشنهادی

> نموداری چهاربخشی با عنوان “Comparison of Scaling Methods”،
> که در هر بخش توزیع داده‌ها قبل و بعد از Scaling نمایش داده می‌شود.

---

### 💬 گفت‌وگوی استاد و دانشجو

👩‍💻 استاد، من همیشه از StandardScaler استفاده کنم؟
👨‍🏫 نه عزیزم! اگه داده‌هات نویز زیاد دارن، برو سراغ RobustScaler.
👩‍💻 و اگه می‌خوام بازه‌ی مقادیر رو محدود کنم؟
👨‍🏫 اون موقع MinMaxScaler بهترین گزینه‌ست 🔥

---

### 🧩 تمرین

۱️⃣ دیتاستی با سه ویژگی بساز: یکی نرمال، یکی دارای مقادیر منفی، و یکی دارای Outlier.
۲️⃣ هر سه روش `StandardScaler`, `MinMaxScaler`, `RobustScaler` رو روش اعمال کن.
۳️⃣ نمودار مقایسه‌ای قبل و بعد از Scaling رسم کن.

---

### ❓ پرسش چهارگزینه‌ای

کدام‌یک از روش‌های زیر در برابر داده‌های پرت (Outlier) مقاوم‌تر است؟
A) StandardScaler
B) MinMaxScaler
C) RobustScaler ✅
D) MaxAbsScaler

---
