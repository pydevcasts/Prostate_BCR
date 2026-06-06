
## ⚖️ فصل ۶: مقیاس‌سازی و نرمال‌سازی داده‌ها (Data Scaling & Normalization)

### 📄 صفحه ۴ از ۵ — کاربردها و مقایسه‌ی Scaling و Normalization در مدل‌های واقعی

✍️ *نویسنده: سیامک عباس‌نژاد*
🌐 *[https://github.com/pydevcasts](https://github.com/pydevcasts)*

---

### 🎯 هدف این صفحه

در این بخش یاد می‌گیری:
✅ کِی باید داده‌هاتو Scale کنی و کِی Normalize
✅ تأثیر این دو روش بر مدل‌های مختلف یادگیری ماشین
✅ و چند مثال واقعی از نتایج بهتر بعد از نرمال‌سازی یا مقیاس‌دهی 📈

---

### 💡 تفاوت در عمل: Scaling vs Normalization

بذار یه مثال واقعی ببینیم 👇

فرض کن می‌خوای **فاصله‌ی بین افراد** رو بر اساس دو ویژگی بسنجی:

* قد (Height) → بین 150 تا 200
* درآمد (Income) → بین 2,000,000 تا 100,000,000

اگر داده‌ها رو همون‌طور خام بدی به مدل، درآمد چون مقیاس بزرگ‌تری داره، کل فاصله‌ها رو تحت تأثیر قرار می‌ده 💸
در نتیجه، مدل تصور می‌کنه درآمد خیلی مهم‌تر از قد هست، در حالی که شاید اینطور نباشه!

✅ با **Scaling** این مشکل برطرف می‌شه.

---

### 🧩 مقایسه با مثال واقعی

فرض کن داده‌ای داری برای پیش‌بینی احتمال خرید مشتری:

| مشتری | درآمد (میلیون) | قد (سانتی‌متر) | خرید کرده؟ |
| :---- | :------------: | :------------: | :--------: |
| A     |        5       |       170      |     بله    |
| B     |       100      |       165      |     خیر    |
| C     |       10       |       180      |     بله    |
| D     |       50       |       175      |     خیر    |

حالا مدل KNN رو روی داده‌ی **بدون Scaling** و **با Scaling** اجرا می‌کنیم 👇

---

### 💻 کد پایتون مقایسه‌ای

```python
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.DataFrame({
    'Income': [5, 100, 10, 50],
    'Height': [170, 165, 180, 175],
    'Buy': ['Yes', 'No', 'Yes', 'No']
})

X = data[['Income', 'Height']]
y = data['Buy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# مدل بدون Scaling
model_raw = KNeighborsClassifier(n_neighbors=1)
model_raw.fit(X_train, y_train)
print("بدون Scaling:", model_raw.predict(X_test).tolist())

# مدل با Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = KNeighborsClassifier(n_neighbors=1)
model_scaled.fit(X_train_scaled, y_train)
print("با Scaling:", model_scaled.predict(X_test_scaled).tolist())
```

📊 نتیجه:

```
بدون Scaling: ['No', 'No']
با Scaling: ['Yes', 'No']
```

⚡ نتیجه با Scaling دقیق‌تر بود چون حالا مدل فاصله‌ی واقعی بین ویژگی‌ها رو به‌درستی سنجید.

---

### 🤖 تأثیر بر مدل‌های مختلف

| نوع مدل             | نیاز به Scaling | نیاز به Normalization | توضیح                                    |
| :------------------ | :-------------: | :-------------------: | :--------------------------------------- |
| Linear Regression   |      ✅ بله      |          ❌ نه         | برای پایداری وزن‌ها                      |
| Logistic Regression |      ✅ بله      |          ❌ نه         | برای همگرایی سریع‌تر                     |
| KNN                 |      ✅ بله      |         ✅ بله         | چون مبتنی بر فاصله است                   |
| SVM                 |      ✅ بله      |         ✅ بله         | چون بر پایه‌ی فواصل و بردارها کار می‌کند |
| Decision Tree       |       ❌ نه      |          ❌ نه         | چون مبتنی بر مقیاس نیست                  |
| Neural Networks     |      ✅ بله      |         ✅ بله         | برای پایداری گرادیان‌ها                  |
| Naive Bayes         |       ❌ نه      |          ❌ نه         | از احتمال‌ها استفاده می‌کند، نه فاصله‌ها |

---

### 📊 نکته حرفه‌ای

> همیشه قبل از آموزش مدل، با `scaler.fit_transform(X_train)` داده‌های آموزش را مقیاس‌بندی کن
> و برای داده‌های تست از `scaler.transform(X_test)` استفاده کن —
> چون داده‌های تست نباید در آموزش دخالت داشته باشن 🚫

---

### 🎨 تصویر پیشنهادی

> نموداری با دو محور ویژگی (مثلاً Income و Height) که
> قبل از Scaling داده‌ها در گوشه‌ی نمودار جمع شده‌اند و بعد از Scaling به‌شکل یکنواخت و متوازن پخش شده‌اند.

---

### 💬 گفت‌وگوی استاد و دانشجو

👩‍💻 استاد، چرا درخت تصمیم (Decision Tree) نیاز به Scaling نداره؟
👨‍🏫 چون درخت فقط مقایسه‌ی مقادیر انجام می‌ده، مثلاً «آیا X > ۵۰؟»
👩‍💻 پس فاصله براش مهم نیست؟
👨‍🏫 دقیقاً، واسه همین بدون Scaling هم خوب کار می‌کنه 🌳

---

### 🧩 تمرین

۱️⃣ دیتاست `Iris` رو از scikit-learn بارگذاری کن.
۲️⃣ یک بار مدل SVM رو بدون Scaling آموزش بده و دقتش رو محاسبه کن.
۳️⃣ سپس داده‌ها رو با `StandardScaler` نرمال کن و دقت رو دوباره اندازه بگیر.
۴️⃣ نتایج رو مقایسه کن — تفاوت احتمالاً چشمگیره 👀

---

### ❓ پرسش چهارگزینه‌ای

کدام مدل بیشترین حساسیت را نسبت به مقیاس ویژگی‌ها دارد؟

A) Decision Tree

B) K-Nearest Neighbors ✅

C) Naive Bayes

D) Random Forest

---

