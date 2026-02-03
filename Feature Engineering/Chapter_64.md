

# 🚀 فصل ۱۳ — Encoding

## 📄 صفحه ۵ از ۶ — Advanced Encoding

### (Hashing Encoding + Leave-One-Out + CatBoost Encoding)

✍️ *سیامک عباس‌نژاد*
🌐 *[https://github.com/pydevcasts](https://github.com/pydevcasts)*

---

# 🎯 هدف این صفحه

در این صفحه یاد می‌گیری:

✔️ Hashing Encoding چیست و چه زمانی بهترین انتخاب است
✔️ Leave-One-Out Encoding و چرا بسیار خاص است
✔️ CatBoost Encoding (بهترین روش برای مدل‌های درختی)
✔️ مثال عددی + کد پایتون
✔️ خطرات و نکات خیلی حرفه‌ای
✔️ آزمون چهارگزینه‌ای

---

# 🟣 ۱. Hashing Encoding

Hashing یکی از سریع‌ترین روش‌های Encoding است که:

* تعداد دسته‌ها خیلی زیاد باشد
* One-Hot ترکیده باشد
* Target Encoding باعث Leakage شود
* یا مدل باید خیلی بزرگ و real-time باشد

✔️ هیچ Look-up جدول لازم ندارد
✔️ در سیستم‌های Big Data (Spark, Kafka, Online ML) عالی است

---

## 🔵 چگونه کار می‌کند؟

هر دسته را از طریق یک تابع Hash به یک عدد ثابت و محدود تبدیل می‌کند.

مثال:

فرض کن ۵۰۰هزار دسته داریم
ولی ما فقط ۵۰ ستون می‌خواهیم.

Hashing همه آنها را به ۵۰ ستون نگاشت می‌کند.

---

## ⚠️ مشکل اصلی

Collision
یعنی چند دسته در یک ستون هش شوند.

اما در کاربردهای بزرگ این طبیعی و قابل قبول است.

---

## 💻 کد پایتون — Hashing Encoding

(با راست‌چین و بدون شروع خط با حرف انگلیسی)

```python
from sklearn.feature_extraction import FeatureHasher
import pandas as pd

df = pd.DataFrame({
    "city": ["Tehran", "Tabriz", "Shiraz", "Tehran"]
})

h = FeatureHasher(n_features=5, input_type='string')
hashed = h.transform(df["city"])
hashed_df = pd.DataFrame(hashed.toarray())

print(hashed_df)
```

---

# 🟣 ۲. Leave-One-Out Encoding

این روش بسیار شبیه Target Encoding است
اما یک تفاوت طلایی دارد:

✔️ میانگین Target همان دسته **بدون** دادهٔ فعلی محاسبه می‌شود.

به همین دلیل اسمش Leave-One-Out است.

---

## 🔥 مثال

داده:

| City   | Salary |
| ------ | ------ |
| Tehran | 50     |
| Tehran | 60     |
| Tehran | 70     |

برای ردیف اول:

```
(60 + 70) / 2 = 65
```

برای ردیف دوم:

```
(50 + 70) / 2 = 60
```

برای ردیف سوم:

```
(50 + 60) / 2 = 55
```

این باعث می‌شود **هیچ ردیفی از مقدار خودش استفاده نکند**
و Leakage تقریباً صفر شود.

---

## مزایا

✔️ دقیق‌تر از Target Encoding
✔️ بسیار مناسب برای داده‌های کوچک
✔️ مناسب برای مدل‌های حساس مثل Logistic Regression

---

## معایب

❌ کمی کندتر
❌ اگر دسته فقط یک مقدار داشته باشد → مقدار میانگین کلی جایگزین می‌شود

---

## 💻 کد پایتون — Leave-One-Out

```python
import pandas as pd

df = pd.DataFrame({
    "city": ["Tehran", "Tehran", "Tehran"],
    "salary": [50, 60, 70]
})

encoded = []
for i in range(len(df)):
    temp = df.drop(i)
    mean_val = temp[temp["city"] == df.loc[i, "city"]]["salary"].mean()
    encoded.append(mean_val)

df["city_encoded"] = encoded
print(df)
```

---

# 🟣 ۳. CatBoost Encoding

پیشرفته‌ترین و امن‌ترین نوع Target Encoding
که در الگوریتم CatBoost معرفی شده است.

مزایا:

✔️ ترتیب تصادفی داده‌ها را رعایت می‌کند
✔️ از Leave-One-Out در هر مرحله استفاده می‌کند
✔️ Data Leakage تقریباً صفر
✔️ برای داده‌های خیلی بزرگ عالی است
✔️ در مدل‌های درختی بهترین انتخاب است

CatBoost به دنبال خودکارسازی این روش، آن را تبدیل به استاندارد صنعتی کرد.

---

## CatBoost Encoding چگونه کار می‌کند؟

برای هر ردیف:

* مقدار Target دسته را تا همان لحظه محاسبه می‌کند
* ردیف فعلی را حساب نمی‌کند
* ترکیبی از Prior + Mean دسته + Noise استفاده می‌کند

این یعنی:

🔥 *هوشمندترین شکل Target Encoding جهان.*

---

## 💻 کد پایتون — CatBoost Encoder

```python
from category_encoders import CatBoostEncoder
import pandas as pd

df = pd.DataFrame({
    "city": ["Tehran", "Tabriz", "Shiraz", "Tehran"],
    "salary": [50, 40, 60, 70]
})

enc = CatBoostEncoder(cols=["city"])
df["city_encoded"] = enc.fit_transform(df["city"], df["salary"])

print(df)
```

---

# 🔥 نکات فوق‌حرفه‌ای (Industrial)

### ✔️ زمانی که دسته‌ها بسیار زیاد هستند

Hashing Encoding بهترین انتخاب است.

### ✔️ زمانی که مدل حساس است و داده کم است

Leave-One-Out بهترین انتخاب است.

### ✔️ زمانی که مدل درختی هست

CatBoost Encoding بهترین انتخاب است.

### ✔️ زمانی که دسته‌ها خیلی نامتعادل هستند

CatBoost + Smoothing بهترین جواب را می‌دهد.

### ✔️ زمانی که زمان real-time مهم است

Hashing Encoding بهترین انتخاب است.

---

# 🎨 تصویر پیشنهادی

> یک نمودار که Hashing → فشرده‌سازی دسته‌ها
> Leave-One-Out → محاسبه میانگین بدون ردیف جاری
> CatBoost → محاسبه هدف به‌صورت ترتیبی
> را نشان دهد.

---

# 🧠 تمرین

یک ستون شامل ۲۰۰ دسته بساز
سه نسخه از آن را با روش‌های زیر Encode کن:

* Hashing
* Leave-One-Out
* CatBoost Encoding

و سپس:

* یک مدل Logistic Regression
* و یک مدل RandomForest

روی هر سه نسخه تست کن.
نتایج را مقایسه کن.

---

# ❓ آزمون چهارگزینه‌ای

**کدام روش برای دسته‌های بسیار زیاد (۵۰هزار+) مناسب‌تر است؟**

A) One-Hot
B) Label Encoding
C) Hashing Encoding   ✅
D) Ordinal Encoding

---
