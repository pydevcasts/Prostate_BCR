
# 🚀 فصل ۱۳ — Encoding

## 📄 صفحه ۲ از ۶ — One-Hot & Label Encoding به‌صورت کامل و حرفه‌ای

✍️ *سیامک عباس‌نژاد*
🌐 *[https://github.com/pydevcasts](https://github.com/pydevcasts)*

---

# 🎯 هدف این صفحه

در این صفحه یاد می‌گیری:

✅ Label Encoding چیست و *چه زمانی* باید استفاده شود
✅ One-Hot Encoding چیست و *کجا خطرناک می‌شود*
✅ تفاوت‌های کلیدی این دو
✅ نکات حرفه‌ای برای پروژه‌های واقعی
✅ مثال عددی + کد پایتون
✅ خطاهای رایج دانشجوها

---

## 🔵 ۱. چیست Label Encoding

یعنی تبدیل دسته‌ها به اعداد ترتیبی.
مثال:

```
Tehran → 0
Tabriz → 1
Shiraz → 2
```

## 🟣 کاربرد درست

✔️ فقط برای داده‌های *Ordinal* استفاده شود:

* اندازه لباس: S < M < L < XL
* سطح رضایت: Poor < Fair < Good < Excellent
* تحصیلات: Diploma < Bachelor < Master < PhD

چون ترتیب واقعی در این داده‌ها وجود دارد.

## 🔥 کاربرد غلط

❌ برای دسته‌های Nominal مثل شهر، محصول، رنگ
چون مدل فکر می‌کنه:

```
Tehran < Tabriz < Shiraz
```

که معنا ندارد.

---

## 🔵 ۲.چیست  One-Hot Encoding

این روش برای *هر دسته یک ستون جدید* ایجاد می‌کند:

| city   | city_Tehran | city_Tabriz | city_Shiraz |
| ------ | ----------- | ----------- | ----------- |
| Tehran | 1           | 0           | 0           |
| Tabriz | 0           | 1           | 0           |
| Shiraz | 0           | 0           | 1           |

## 🟣 کاربرد درست

✔️ برای داده‌های Nominal
✔️ برای داده‌هایی با تعداد دسته *کم*
✔️ برای مدل‌هایی مثل Logistic Regression, Linear مدل‌ها

## ⚠️ کاربرد غلط

❌ اگر تعداد دسته‌ها زیاد باشد:
مثلاً ۱۰٬۰۰۰ شهر → ۱۰٬۰۰۰ ستون! → مدل نابود می‌شود.

❌ اگر دسته‌های نادر زیاد باشد
One-Hot فضا را sparse و مدل را ضعیف می‌کند.

---

## 🔥 ۳. مقایسهٔ حرفه‌ای Label vs One-Hot

| ویژگی                  | Label Encoding | One-Hot Encoding |
| ---------------------- | -------------- | ---------------- |
| نوع داده مناسب         | Ordinal        | Nominal          |
| تولید ترتیب مصنوعی     | بله ❌          | خیر ✔️           |
| افزایش ابعاد           | کم ✔️          | زیاد ❌           |
| مناسب برای مدل‌های خطی | خیر            | بله ✔️           |
| مناسب برای درخت‌ها     | بله ✔️         | بله ✔️           |
| ریسک Overfitting       | کم             | زیاد             |

---

## 🧪 ۴. مثال عددی مهم

فرض کن ستون “رنگ” داریم:

| Color |
| ----- |
| Red   |
| Green |
| Blue  |
| Red   |

### Label Encoding اشتباه:

```
Blue → 0
Green → 1
Red → 2
```

مدل حالا فکر می‌کند:

```
Red > Green > Blue
```

که کاملاً غلط است.

### One-Hot صحیح:

```
Red → 1 0 0
Green → 0 1 0
Blue → 0 0 1
```

---

## 💻 ۵. کد پایتون — Label + One-Hot

```python
import pandas as pd
# نمونه Label و One-Hot Encoding

data = pd.DataFrame({
    "color": ["Red", "Green", "Blue", "Red"]
})

# Label Encoding
data["label_enc"] = data["color"].astype("category").cat.codes

# One-Hot Encoding
onehot = pd.get_dummies(data["color"], prefix="color")
data = pd.concat([data, onehot], axis=1)

print(data)
```

---

## ⚠️ ۶. نکات حرفه‌ای (برای Kaggle و پروژه واقعی)

### ✔️ نکته ۱ — همیشه One-Hot را فقط روی دادهٔ Train fit کن

اگر روی test هم fit کنی → نشت اطلاعات (Data Leakage)

### ✔️ نکته ۲ — دسته‌های جدید در دادهٔ Test

مثلاً در Train فقط ۳ رنگ داریم
اما در Test رنگ جدید "Yellow" آمده
One-Hot → کرش نمی‌کند، ستونش را ندارد → باید مدیریت شود.

### ✔️ نکته ۳ — در مدل‌های درختی، Label Encoding عالی است

در Random Forest / XGBoost ترتیب مصنوعی را یاد نمی‌گیرند، پس مشکلی نیست.

---

## 🎨 ۷. تصویر پیشنهادی

> نموداری که One-Hot را به‌صورت گرافیکی نشان می‌دهد:
> مستطیل‌های جداگانه برای هر دسته، بدون ترتیب
> در مقابل Label که به‌صورت خطی 0،1،2 دیده می‌شود.

---

## 🧠 ۸. تمرین کاربردی

۱) یک دیتاست شامل ۵ ویژگی دسته‌ای انتخاب کن
۲) روی آن‌ها هر دو روش Label + One-Hot اجرا کن
۳) با Logistic Regression تست کن که کدام روش بهتر است
۴) نتیجه را یادداشت کن.

---

## ❓ آزمون چهارگزینه‌ای صفحه

**در کدام حالت Label Encoding انتخاب مناسب‌تری است؟**

A) رنگ خودرو

B) شهر محل تولد

C) اندازه لباس S < M < L < XL ✅

D) نام کشورها

