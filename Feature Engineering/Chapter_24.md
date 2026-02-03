
## 🧠 فصل ۵: انتخاب ویژگی‌ها (Feature Selection)

### 📄 صفحه ۴ از ۵ — روش‌های Embedded (انتخاب ویژگی درون فرآیند آموزش مدل)

✍️ *نویسنده: سیامک عباس‌نژاد*
🌐 *[https://github.com/pydevcasts](https://github.com/pydevcasts)*

---

### 🎯 هدف این صفحه

در این صفحه یاد می‌گیری چطور مدل‌های خاص، در حین یادگیری، خودشون ویژگی‌های مهم رو **وزن‌دهی یا حذف** می‌کنن.
این روش‌ها ترکیبی از دو روش قبلی (Filter + Wrapper) هستن،
و برای داده‌های واقعی **بسیار کارآمد** محسوب می‌شن ⚡

---

### 💡 مفهوم روش‌های Embedded

در روش Embedded، فرآیند انتخاب ویژگی با آموزش مدل **هم‌زمان** انجام می‌شه.
یعنی وقتی مدل یاد می‌گیره، درون خودش تصمیم می‌گیره که وزن هر ویژگی چقدره
و در نهایت ویژگی‌های با اهمیت کم رو کنار می‌گذاره ✂️

📘 مثال:
در رگرسیون Lasso، مدل به ویژگی‌هایی که بی‌اهمیت هستن وزن صفر می‌ده.
پس نیازی به حذف دستی اون‌ها نیست 😎

---

### ⚙️ مزایا و معایب

| مزایا                         | معایب                                                          |
| :---------------------------- | :------------------------------------------------------------- |
| ⚡ سریع‌تر از Wrapper          | ❌ فقط در مدل‌هایی قابل‌اجراست که وزن ویژگی‌ها رو محاسبه می‌کنن |
| 🎯 دقت بالا و قابل تفسیر      | ⚠️ نیاز به تنظیم دقیق پارامترها                                |
| 💡 انتخاب خودکار در طول آموزش | —                                                              |

---

### 🔍 مدل‌های معروف Embedded

| مدل                                               | روش انتخاب ویژگی                         | توضیح                             |
| :------------------------------------------------ | :--------------------------------------- | :-------------------------------- |
| 🧮 **Lasso Regression (L1 Regularization)**       | حذف خودکار ویژگی‌های با وزن صفر          | مناسب برای داده‌های عددی          |
| ⚙️ **Tree-based Models (Random Forest, XGBoost)** | بر اساس اهمیت ویژگی (Feature Importance) | بسیار محبوب در دیتاست‌های واقعی   |
| 🧠 **ElasticNet**                                 | ترکیب L1 و L2 Regularization             | تعادل بین حذف و کاهش وزن ویژگی‌ها |

---

### 💻 مثال ۱: Lasso Regression

در رگرسیون Lasso، مدل برای کاهش پیچیدگی، وزن بعضی ویژگی‌ها را صفر می‌کند.

```python
from sklearn.linear_model import Lasso
from sklearn.datasets import load_diabetes
import pandas as pd

data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

lasso = Lasso(alpha=0.05)
lasso.fit(X, y)

importance = pd.Series(lasso.coef_, index=X.columns)
print(importance.sort_values(ascending=False))
```

📊 خروجی نمونه:

```
bmi      500.2
bp       210.8
s5        87.5
s3       -12.3
s1        0.0
s6        0.0
```

💡 ویژگی‌هایی که مقدار صفر دارن یعنی از دید مدل **غیرمفید** بودن و حذف می‌شن ✅

---

### 💻 مثال ۲: Random Forest Feature Importance

در مدل‌های درختی مثل RandomForest، خود مدل بر اساس **کاهش خطای گینی (Gini)** یا **Entropy**،
به‌صورت خودکار اهمیت هر ویژگی رو محاسبه می‌کنه 🌲

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False))
```

📊 خروجی نمونه:

```
petal length (cm)    0.43
petal width (cm)     0.39
sepal length (cm)    0.12
sepal width (cm)     0.06
```

💡 یعنی دو ویژگی اول بیشترین تأثیر رو روی پیش‌بینی نوع گل داشتن 🌸

---

### 🎨 تصویر پیشنهادی

> نموداری میله‌ای (Bar Chart) از اهمیت ویژگی‌ها در مدل RandomForest
> که در آن دو ستون بزرگ‌تر از بقیه‌اند و رنگ طلایی دارند.

---

### 💬 گفت‌وگوی استاد و دانشجو

👩‍💻 استاد، پس این روش خودش ویژگی‌ها رو حذف می‌کنه؟
👨‍🏫 دقیقاً! Embedded یعنی مدل در حین یادگیری، ویژگی‌ها رو قضاوت می‌کنه و خودش تصمیم می‌گیره.
👩‍💻 یعنی نیاز نیست دستی حذف کنم؟
👨‍🏫 بله، فقط باید پارامترهای تنظیمی مثل `alpha` در Lasso رو درست انتخاب کنی تا تعادل برقرار شه ⚖️

---

### 🧩 تمرین

۱️⃣ دیتاستی از فروش محصولات بساز با ستون‌های `price`, `discount`, `views`, `sales`.
۲️⃣ با مدل `Lasso(alpha=0.1)` یادگیری کن.
۳️⃣ بررسی کن کدام ویژگی‌ها وزن صفر گرفته‌اند.
۴️⃣ ویژگی‌های غیرمفید را حذف کن و دوباره مدل را آموزش بده.

---

### ❓ پرسش چهارگزینه‌ای

کدام گزینه درباره‌ی روش‌های Embedded درست است؟
A) از مدل برای انتخاب ویژگی استفاده نمی‌کنند
B) ویژگی‌ها را به‌صورت تصادفی حذف می‌کنند
C) در حین آموزش مدل، ویژگی‌های مهم را شناسایی می‌کنند ✅
D) فقط برای داده‌های متنی مناسب‌اند

---
