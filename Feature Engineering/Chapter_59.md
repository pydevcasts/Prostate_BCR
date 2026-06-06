

# ⚙️ فصل ۱۲ — ساخت ویژگی‌های جدید

## 📄 صفحه ۵ از ۵ — تکنیک‌های پیشرفته ساخت ویژگی (Advanced Feature Construction)

✍️ *نویسنده: سیامک عباس‌نژاد*
🌐 *[https://github.com/pydevcasts](https://github.com/pydevcasts)*

---

## 🎯 هدف این صفحه

در پایان این بخش تو یاد می‌گیری:
✅ Binning (گروه‌بندی عددی)
✅ Rank Features
✅ Ratio Features (نسبت‌ها)
✅ Feature Crossing (ترکیب پیشرفته ویژگی‌ها)
✅ Statistical Encoding
✅ Target-based Construction
✅ Count & Frequency Encoding
به همراه مثال عددی + کد پایتون.

---

## 🧩 ۱. گروه‌بندی عددی — Binning 

گاهی ویژگی عددی خیلی خام است.
مثال ساده:

```
Age = 1, 2, 3, ..., 90
```

ولی مدل لازم ندارد تک‌تک این‌ها را بداند.
می‌توانیم آن را به گروه تبدیل کنیم:

| سن | گروه    |
| -- | ------- |
| 2  | کودک    |
| 15 | نوجوان  |
| 32 | بزرگسال |
| 70 | سالمند  |

این کار:
✨ نویز را کاهش می‌دهد
✨ مدل را پایدارتر می‌کند
✨ به توزیع بهتر کمک می‌کند

پایتون:

```python
bins = [0, 12, 20, 40, 60, 100]
# گروه‌بندی سن
labels = ["kid", "teen", "adult", "mid_age", "senior"]

data["age_group"] = pd.cut(data["age"], bins=bins, labels=labels)
```

---

## 🔢 ۲. رتبه ویژگی Rank Features

در بسیاری از پروژه‌ها *مقدار مطلق مهم نیست* → رتبه مهم است.

نمونه:

| درآمد | رتبه درآمد |
| ----- | ---------- |
| 1200  | 1          |
| 4000  | 2          |
| 6500  | 3          |

رتبه به ویژه در:

* قیمت‌گذاری
* سیستم‌های امتیازدهی
* مدل‌های مالی
  بسیار مهم است.

مثال پایتون:

```python

data["income_rank"] = data["income"].rank()
# ساخت رتبه
```

---

## ➗ ۳. ویژگی‌های نسبت — Ratio Features  

نسبت‌ها یکی از **قوی‌ترین** نوع ویژگی‌ها هستند.

مثال‌ها:

```
income / age
rooms / area
debt / income
price / size
```

این نسبت‌ها معنی‌های جدید می‌سازند
و در تمام پروژه‌های واقعی بانکی، فروش و املاک استفاده می‌شوند.

پایتون:

```python
data["debt_to_income"] = data["debt"] / data["income"]
```

---

## ✖️ ۴. ترکیب ویژگی‌ها — Feature Crossing 

در این روش چند ویژگی را داخل هم ضرب می‌کنیم تا تعامل پیچیده‌تر ساخته شود.

مثال واقعی:

```
Age * CreditScore
Rooms * NeighborhoodQuality
Price_per_unit * Season
```

نمونه پایتون:

```python
data["age_credit_cross"] = data["age"] * data["credit_score"]
```

این روش در سیستم‌های Ranking مثل
YouTube, Google, Recommender Systems
به شدت رایج است.

---

## 📊 ۵. ویژگی‌های آماری روی دسته‌ها  — Statistical Encoding

برای دسته‌های مختلف آماری جدید می‌سازیم.

مثال:

| شهر   | میانگین فروش |
| ----- | ------------ |
| تبریز | 5.2          |
| تهران | 8.7          |
| شیراز | 6.3          |

بعد این مقدار را وارد مدل می‌کنیم.

پایتون:

```python
city_mean = data.groupby("city")["sales"].transform("mean")
data["city_mean_sales"] = city_mean
```

---

## 🔢 ۶. تکرار Frequency & Count Encoding

برای دسته‌هایی با تعداد بالا فوق‌العاده مؤثر است.

مثال:

| محصول | تعداد تکرار |
| ----- | ----------- |
| A     | 120         |
| B     | 20          |
| C     | 5           |

پایتون:

```python
freq = data["product"].value_counts()
data["product_freq"] = data["product"].map(freq)
```

این روش از One-Hot بهتر است وقتی:

* دسته‌ها زیاد هستند
* و (sparse) زیاد تولید نشود
* حافظه کم باشد

---

## 🧠 ۷. (نسخهٔ حرفه‌ای) Target Encoding 

برای هر دسته میانگین مقدار هدف (y) را محاسبه می‌کنیم.

مثال:

| شهر   | Y (فروش) |
| ----- | -------- |
| تهران | 80       |
| تبریز | 40       |

در مدل‌های Kaggle این تکنیک خیلی کاربرد دارد.

پایتون:

```python
target_mean = data.groupby("city")["sales"].transform("mean")
data["city_target_enc"] = target_mean
```

توجه: باید با Cross-Validation انجام شود که نشت اطلاعات نشود.

---

## 🧮 ۸. کد پایتون — ترکیب همهٔ روش‌های پیشرفته

```python
import pandas as pd
# ترکیب چند تکنیک پیشرفته

data = pd.DataFrame({
    "age": [10, 20, 35, 50, 70],
    "income": [200, 1500, 4000, 9000, 12000],
    "city": ["Tabriz", "Tehran", "Tabriz", "Shiraz", "Tehran"],
    "debt": [0, 200, 600, 1500, 2500]
})

# Binning
bins = [0, 18, 40, 60, 100]
labels = ["young", "adult", "mid", "old"]
data["age_group"] = pd.cut(data["age"], bins=bins, labels=labels)

# Ratio
data["debt_to_income"] = data["debt"] / data["income"]

# Rank
data["income_rank"] = data["income"].rank()

# Frequency Encoding
freq = data["city"].value_counts()
data["city_freq"] = data["city"].map(freq)

# Statistical Encoding
data["city_mean_income"] = data.groupby("city")["income"].transform("mean")

print(data)
```

---

## 🎨 ۹. تصویر پیشنهادی

> نمودار Heatmap از Feature Crossings
> برای نشان دادن تعامل ویژگی‌های جدید

---

# 🧠 ۱۰. تمرین عملی

۱) یک دیتاست واقعی انتخاب کن
۲) ۱۰ ویژگی جدید از نوع‌های مختلف (Ratio, Count, Rank, Cross) بساز
۳) قبل و بعد مدل را اجرا کن
۴) Feature Importance را بررسی کن
۵) تحلیل کن کدام ویژگی بیشترین اثر را داشت

---

# ❓ آزمون چهارگزینه‌ای صفحه

**کدام روش برای ویژگی‌های دسته‌ای که تعداد زیادی سطح (Category) دارند مناسب‌تر است؟**

A) One-Hot Encoding

B) Frequency Encoding ✅

C) Standardization

D) Polynomial Features

