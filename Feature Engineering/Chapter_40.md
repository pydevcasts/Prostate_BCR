

## 🧩 فصل ۸: ایجاد ویژگی‌های جدید (Feature Construction & Generation)

### 📄 صفحه ۵ از ۵ — ایجاد ویژگی‌های ترکیبی و تعاملی (Feature Interaction & Cross Features)

✍️ *نویسنده: سیامک عباس‌نژاد*
🌐 *[https://github.com/pydevcasts](https://github.com/pydevcasts)*

---

### 🎯 هدف این صفحه

در این بخش یاد می‌گیری:
✅ چطور ویژگی‌های موجود رو با هم ترکیب کنی
✅ از ضرب، نسبت و تفاوت بین متغیرها استفاده کنی
✅ چطور تعامل بین ویژگی‌ها رو در مدل لحاظ کنی
✅ و با مثال‌ها و کدهای پایتون این مفاهیم رو تمرین کنی 💻

---

## 🧠 ۱. چرا ویژگی‌های ترکیبی مهم‌اند؟

گاهی مدل، ارتباط بین دو ویژگی رو به‌صورت مستقیم درک نمی‌کنه.
مثلاً فقط با دانستن «تعداد خرید» و «قیمت هر خرید»، مدل نمی‌فهمه «کل درآمد» چقدره.
اما اگه ویژگی جدیدی بسازی به‌نام:

> **Total_Revenue = Quantity × Price**

مدل یک مفهوم اقتصادی جدید یاد می‌گیره 💰

---

### 💡 مثال عددی ساده

| Quantity | Price | Total_Revenue |
| :------: | :---: | :-----------: |
|     2    |  100  |      200      |
|     4    |   50  |      200      |
|    10    |   30  |      300      |

مدل حالا می‌تونه بفهمه دو مشتری که خریدهای متفاوت داشتن، درآمد مشابهی ایجاد کردن.

---

## ⚙️ ۲. انواع ویژگی‌های ترکیبی

| نوع ترکیب                | فرمول                | کاربرد                  |
| :----------------------- | :------------------- | :---------------------- |
| **جمع (Sum)**            | `A + B`              | ادغام دو شاخص هم‌جهت    |
| **تفاضل (Difference)**   | `A - B`              | سنجش اختلاف دو متغیر    |
| **نسبت (Ratio)**         | `A / B`              | مقایسه نسبی دو ویژگی    |
| **ضرب (Multiplication)** | `A * B`              | ساخت قدرت تعاملی        |
| **Cross Feature**        | ترکیب دو متغیر گسسته | نمایش تعامل بین گروه‌ها |

---

## 💻 ۳. مثال پایتون – ساخت ویژگی‌های ترکیبی

```python
import pandas as pd

data = pd.DataFrame({
    'Quantity': [2, 4, 10],
    'Price': [100, 50, 30],
    'Discount': [5, 10, 0]
})

data['Total_Revenue'] = data['Quantity'] * data['Price']
data['Revenue_After_Discount'] = data['Total_Revenue'] * (1 - data['Discount'] / 100)
data['Price_Per_Unit'] = data['Price'] / data['Quantity']

print(data)
```

📊 **خروجی:**

```
   Quantity  Price  Discount  Total_Revenue  Revenue_After_Discount  Price_Per_Unit
0         2    100         5            200                 190.0            50.0
1         4     50        10            200                 180.0            12.5
2        10     30         0            300                 300.0             3.0
```

✅ حالا مدل درک می‌کنه که تخفیف روی درآمد تأثیر مستقیم داره و واحد قیمت برای مقایسه‌ی مشتریان مفیده.

---

## 🎯 ۴. ویژگی‌های تعاملی (Interaction Features)

گاهی دو ویژگی در تعامل با هم اثر دارن.
مثلاً **تأثیر تبلیغات (Ad Spend)** بستگی به **فصل سال (Season)** داره.
در این حالت، تعاملشون یعنی:

> **Ad_Spend × Season**

یا برای ویژگی‌های متنی / دسته‌ای، ترکیبشون رو با هم می‌سازیم:

| Ad_Channel | Region | Combined    |
| :--------- | :----- | :---------- |
| TV         | North  | TV_North    |
| Online     | West   | Online_West |

---

### 💻 مثال پایتون با داده‌های دسته‌ای

```python
data = pd.DataFrame({
    'Ad_Channel': ['TV', 'Online', 'TV', 'Online'],
    'Region': ['North', 'West', 'South', 'West']
})

data['Cross_Feature'] = data['Ad_Channel'] + '_' + data['Region']
print(data)
```

📊 **خروجی:**

```
  Ad_Channel Region Cross_Feature
0         TV  North      TV_North
1     Online   West   Online_West
2         TV  South      TV_South
3     Online   West   Online_West
```

✅ مدل حالا می‌تونه تشخیص بده مثلاً تبلیغ تلویزیونی در منطقه‌ی شمال بهتر جواب داده تا جنوب.

---

## 🔬 ۵. ساخت خودکار ویژگی‌های تعاملی

پکیج **PolynomialFeatures** در Scikit-Learn این کار رو خودکار انجام می‌ده.

---

### 💻 مثال پایتون با `PolynomialFeatures`

```python
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

data = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(data)

pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['A', 'B']))
```

📊 **خروجی:**

```
   A  B  A^2  A B  B^2
0  1  4    1   4   16
1  2  5    4  10   25
2  3  6    9  18   36
```

✅ حالا مدل هم ویژگی‌های اصلی (A، B) رو داره، هم تعاملشون (A×B) و توان‌هاشون.

---

## 🎨 تصویر پیشنهادی

> نموداری دوبعدی از داده‌ها با دو ویژگی A و B که سطح تصمیم مدل بدون ویژگی‌های تعاملی خطی است،
> و سپس همان نمودار با ویژگی‌های تعاملی که مرز تصمیم به‌شکل منحنی درمی‌آید.

---

## 💬 گفت‌وگوی استاد و دانشجو

👩‍💻 استاد، یعنی هر چی ویژگی ترکیبی بیشتر بسازم، مدل بهتر می‌شه؟
👨‍🏫 نه دقیقاً. اگر بیش از حد بسازی، مدل بیش‌برازش (Overfitting) می‌کنه!
👩‍💻 پس باید با دقت و منطق انتخابشون کنم؟
👨‍🏫 آفرین 👏 بهتره فقط تعامل‌هایی رو بسازی که از دید دامنه‌ی مسئله منطقی هستن.

---

## 🧩 تمرین

۱️⃣ داده‌ای با ستون‌های `Income`, `Age`, و `Spend` بساز.
۲️⃣ ویژگی‌های جدیدی مثل `Income/Age`, `Income*Spend`, و `Spend/Age` اضافه کن.
۳️⃣ بررسی کن کدام‌یک همبستگی بالاتری با متغیر هدف دارد.

---

## ❓ پرسش چهارگزینه‌ای

کدام گزینه نمونه‌ای از **ویژگی تعاملی (Cross Feature)** است؟
A) `Age`
B) `Salary`
C) `Age * Salary` ✅
D) `Salary / 12`

