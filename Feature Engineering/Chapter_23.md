

## 🧠 فصل ۵: انتخاب ویژگی‌ها (Feature Selection)

### 📄 صفحه ۳ از ۵ — روش‌های Wrapper (انتخاب ویژگی با استفاده از مدل)

✍️ *نویسنده: سیامک عباس‌نژاد*
🌐 *[https://github.com/pydevcasts](https://github.com/pydevcasts)*

---

### 🎯 هدف این صفحه

در این بخش یاد می‌گیری چطور با استفاده از خود **مدل یادگیری ماشین**،
به‌صورت مستقیم تست کنی کدام ترکیب از ویژگی‌ها بهترین عملکرد را دارد.
به این روش‌ها می‌گویند **Wrapper Methods** چون مدل مثل یک “پوشش” روی داده‌ها می‌پیچه تا بفهمه کدوم ویژگی‌ها مفیدترند. 🎯

---

### 💡 مفهوم روش‌های Wrapper

در روش‌های Filter، فقط آمار داده بررسی می‌شد.
اما در **Wrapper**، ما یک مدل انتخاب می‌کنیم (مثلاً Logistic Regression, RandomForest, KNN و...)
و بارها مدل را با ترکیب‌های مختلف از ویژگی‌ها اجرا می‌کنیم تا بهترین مجموعه پیدا شود 🔁

---

### ⚙️ منطق کلی

1️⃣ انتخاب یک زیرمجموعه از ویژگی‌ها
2️⃣ آموزش مدل روی همان زیرمجموعه
3️⃣ محاسبه‌ی دقت (Accuracy) یا معیار خطا
4️⃣ تکرار مراحل بالا برای ترکیب‌های دیگر
5️⃣ انتخاب ترکیبی که بهترین دقت را دارد ✅

---

### 💻 روش‌های معروف Wrapper

| روش                                        | توضیح                                                         | مزایا           | معایب                            |
| :----------------------------------------- | :------------------------------------------------------------ | :-------------- | :------------------------------- |
| 🔼 **Forward Selection**                   | از صفر شروع می‌کند و ویژگی‌ها را یکی‌یکی اضافه می‌کند         | سریع‌تر از بقیه | ممکن است در محلی بهینه متوقف شود |
| 🔽 **Backward Elimination**                | با همه ویژگی‌ها شروع کرده و یکی‌یکی حذف می‌کند                | دقیق‌تر         | کندتر                            |
| 🔁 **Recursive Feature Elimination (RFE)** | بارها مدل را آموزش می‌دهد و کم‌اهمیت‌ترین ویژگی را حذف می‌کند | محبوب و دقیق    | زمان‌بر برای دیتاست‌های بزرگ     |

---

### 💻 مثال: RFE با Logistic Regression

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import pandas as pd

# داده نمونه
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# تعریف مدل و انتخابگر
model = LogisticRegression(max_iter=500)
rfe = RFE(model, n_features_to_select=5)
fit = rfe.fit(X, y)

# ویژگی‌های انتخاب‌شده
selected = X.columns[fit.support_]
print("Selected Features:", selected)
```

📊 خروجی نمونه:

```
Selected Features: ['mean radius', 'mean texture', 'mean smoothness', 'area error', 'worst concavity']
```

💡 یعنی از بین ده‌ها ویژگی فقط ۵تایی که بیشترین تأثیر روی پیش‌بینی سرطان داشتن انتخاب شدن ✅

---

### ⚙️ روش Forward Selection (افزایشی)

در این روش با یک مدل ساده شروع می‌کنیم و هر بار ویژگی‌ای اضافه می‌کنیم که دقت مدل را بیش‌تر کند:

```python
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

sfs = SequentialFeatureSelector(KNeighborsClassifier(n_neighbors=3),
                                k_features=5,
                                forward=True,
                                scoring='accuracy',
                                cv=3)
sfs.fit(X_train, y_train)

print("Selected Features:", sfs.k_feature_names_)
```

💡 ویژگی‌ها به صورت افزایشی انتخاب می‌شن تا جایی که دقت مدل دیگه زیاد تغییر نکنه.

---

### 🎨 تصویر پیشنهادی

> یک نمودار پله‌ای که در محور X تعداد ویژگی‌ها و در محور Y دقت مدل رسم شده است،
> و نقطه‌ای با رنگ متفاوت نشان می‌دهد که در آن تعداد ویژگی‌ها مدل بهترین عملکرد را دارد.

---

### 💬 گفت‌وگوی استاد و دانشجو

👩‍💻 استاد، یعنی Wrapper همیشه از Filter بهتره؟
👨‍🏫 از نظر دقت بله، چون مدل خودش تصمیم می‌گیره. ولی از نظر سرعت نه! چون ممکنه صدها بار آموزش داده بشه 😅
👩‍💻 پس برای داده‌های بزرگ خطرناک می‌شه؟
👨‍🏫 دقیقاً! برای اون‌ها معمولاً از **Filter یا Embedded** استفاده می‌کنیم ⚡

---

### 🧩 تمرین

یک دیتاست فرضی با ۱۰ ویژگی عددی بساز.
سپس با استفاده از `RFE` و مدل `RandomForestClassifier` فقط ۴ ویژگی برتر را انتخاب کن.
در انتها بررسی کن آیا دقت مدل نسبت به قبل از انتخاب ویژگی‌ها افزایش یافته است یا خیر؟

---

### ❓ پرسش چهارگزینه‌ای

کدام‌یک از گزینه‌ها در مورد روش‌های Wrapper درست است؟
A) از مدل برای ارزیابی اهمیت ویژگی‌ها استفاده می‌کنند ✅
B) سریع‌تر از روش‌های Filter هستند
C) فقط برای داده‌های متنی مناسب‌اند
D) نیازی به ارزیابی دقت مدل ندارند

---
