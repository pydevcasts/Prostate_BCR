
# 🚀 فصل ۱۳ — Encoding تکنیک‌های تبدیل داده‌های دسته‌ای (Categorical Encoding)

✍️ *نویسنده: سیامک عباس‌نژاد*
🌐 *[https://github.com/pydevcasts](https://github.com/pydevcasts)*

---

## 📄 صفحه ۱ از ۶ — مقدمهٔ کامل Encoding و چرا برای مدل ضروری است

*(صفحه ۱ همیشه مقدمه و مفاهیم پایه است)*

---

## 🎯 هدف این صفحه

در پایان این صفحه یاد می‌گیری:
✅ Encoding چیست؟
✅ چرا مدل‌ها دادهٔ دسته‌ای را نمی‌فهمند؟
✅ انواع کلی Encoding (سریع، دقیق، حرفه‌ای)
✅ تفاوت **Nominal** و **Ordinal**
✅ خطاهای رایج Encoding
این پایه برای صفحات بعدی ضروری است.

---

## 🧠 ۱. مفهوم Encoding

یعنی **تبدیل داده‌های غیرعددی (مثل شهر، رنگ، جنسیت، محصول)** به دادهٔ عددی که مدل بتواند آن را بفهمد.

مثال:

```
City:
Tehran
Tabriz
Shiraz
```

مدل‌ها این‌ها را نمی‌فهمند.
باید تبدیل به عدد شوند:

```
Tehran → 0
Tabriz → 1
Shiraz → 2
```

اما!
این روش اشتباه است…
چرا؟
چون مدل فکر می‌کند:

```
Tehran < Tabriz < Shiraz !!!
```

درحالی‌که این ترتیب *نداریم*.

پس Encoding درست نیاز است.

---

## 🎨 ۲. چرا مدل به Encoding نیاز دارد؟

مدل‌های زیر **اجباری** نیاز به Encoding دارند:

* Logistic Regression
* Linear Models
* SVM
* XGBoost
* Random Forest
* KNN

اما مدل‌های Deep Learning مثل Embedding Layer به Encoding خاص می‌پردازند.

---

## 🧩 ۳.(دو نوع اصلی) Categorical Feature Types 

## 🔵  دسته‌ای بدون ترتیب  Nominal

مثال:

* رنگ: قرمز، آبی، سبز
* شهر: تبریز، تهران، شیراز
* جنسیت

هیچ‌کدام ترتیب ندارند → پس نباید به صورت ۰،۱،۲ تبدیل شوند.

---

## 🟣   دسته‌ای با ترتیب Ordinal

مثال:

* کیفیت غذا: بد < متوسط < خوب
* تحصیلات: دیپلم < کارشناسی < ارشد < دکتری
* اندازه لباس: S < M < L < XL

این‌ها ترتیب دارند → پس Encoding ترتیبی مناسب است.

---

## ⚠️ ۴. خطاهای رایج Encoding

❌ اشتباه: Label Encoding برای داده‌های بدون ترتیب
❌ اشتباه: One-Hot برای داده‌هایی با ۵۰۰۰ دسته
❌ اشتباه: Target Encoding بدون Cross-Validation
❌ اشتباه: Encoding در train و test جداگانه انجام نشود

---

## 🔥 ۵. دسته‌بندی انواع Encoding (مهم‌ترین بخش مقدمه)

## ۱)  ساده (سریع و سبک) Encoding

* One-Hot
* Label Encoding
* Ordinal Encoding

مناسب برای داده‌های کوچک.

---

## ۲)  متوسط (دقیق‌تر و هوشمندتر) Encoding

* Frequency Encoding
* Count Encoding
* Hashing Encoding

مناسب وقتی دسته‌ها زیادند.

---

## ۳)  پیشرفته (در Kaggle و صنعت رایج) Encoding

* Target Encoding
* CatBoost Encoding
* GLMM Encoding
* James-Stein Encoding
* MEstimate Encoding

این‌ها دقت مدل را به‌شدت بالا می‌برند (با کنترل نشت اطلاعات).

---

## 🧮 ۶. مثال ساده از Encoding

فرض کنیم ستون City داریم:

| City   |
| ------ |
| Tehran |
| Tabriz |
| Shiraz |

سه روش مختلف:

### ✔️ One-Hot

```
Tehran → 1 0 0
Tabriz → 0 1 0
Shiraz → 0 0 1
```

### ✔️ Label Encoding

```
Tehran → 0
Tabriz → 1
Shiraz → 2
```

⚠️ اشتباه برای داده‌های بدون ترتیب.

### ✔️ Frequency Encoding

```
Tehran → 1000
Tabriz → 300
Shiraz → 200
```

مدل با داده‌های زیاد بهتر کار می‌کند.

---

## 💻 ۷. کد پایتون — نمایش چند Encoding ساده

```python
import pandas as pd
# نمونه Encoding برای یک ستون

data = pd.DataFrame({
    "city": ["Tehran", "Tabriz", "Shiraz", "Tehran"]
})

# Label Encoding
data["label_enc"] = data["city"].astype("category").cat.codes

# One-Hot
onehot = pd.get_dummies(data["city"], prefix="city")
data = pd.concat([data, onehot], axis=1)

# Frequency Encoding
freq = data["city"].value_counts()
data["freq_enc"] = data["city"].map(freq)

print(data)
```

---

## 🎨 ۸. تصویر پیشنهادی

> نمودار میله‌ای مقایسه تعداد دسته‌ها قبل/بعد One-Hot Encoding
> یا نمودار تفاوت Label vs One-Hot vs Frequency

---

## 🧠 ۹. تمرین ساده

۱) یک ستون دسته‌ای انتخاب کن (شهر، رنگ یا محصول)

۲) سه نوع Encoding روی آن اعمال کن:

* Label
* One-Hot
* Frequency

  ۳) نتایج را مقایسه کن

  ۴) بررسی کن کدام روش دقت مدل را بهتر کرد

---

## ❓ آزمون چهارگزینه‌ای صفحه

**کدام روش برای داده‌های بدون ترتیب و تعداد دسته کم مناسب‌تر است؟**

A) Label Encoding

B) One-Hot Encoding ✅

C) Frequency Encoding

D) Hashing Encoding

