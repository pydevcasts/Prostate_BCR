

## ⚙️ فصل ۱۱: انتخاب ویژگی‌ها (Feature Selection)

### 📄 صفحه ۳ از ۵ — Wrapper Methods (روش‌های مدل‌محور)

✍️ *نویسنده: سیامک عباس‌نژاد*
🌐 *[https://github.com/pydevcasts](https://github.com/pydevcasts)*

---

### 🎯 هدف این صفحه

✅ درک دقیق مفهوم Wrapper Methods

✅ آشنایی با سه روش اصلی:

* Recursive Feature Elimination (RFE)
* Forward Selection
* Backward Elimination

  ✅ پیاده‌سازی در پایتون

  ✅ بررسی مزایا و معایب این روش‌ها

---

## 🧭 ۱. مفهوم Wrapper Methods

در روش‌های **Wrapper**، انتخاب ویژگی‌ها بر اساس **عملکرد مدل** انجام می‌شود.
یعنی ویژگی‌ها یکی‌یکی به مدل اضافه یا از آن حذف می‌شوند، و در هر مرحله **دقت مدل** بررسی می‌شود.

📘 خلاصه:

> ویژگی‌ها مثل بازیکنان یک تیم هستند؛ فقط آن‌هایی که در عملکرد تیم (مدل) تأثیر مثبت دارند، انتخاب می‌شوند ⚽

---

## ⚙️ ۲. انواع روش‌های Wrapper

| روش                                        | توضیح کوتاه                                               |
| :----------------------------------------- | :-------------------------------------------------------- |
| 🔹 **Forward Selection**                   | ویژگی‌ها را از صفر شروع کرده و یکی‌یکی اضافه می‌کند       |
| 🔹 **Backward Elimination**                | همه‌ی ویژگی‌ها را دارد و یکی‌یکی حذف می‌کند               |
| 🔹 **RFE (Recursive Feature Elimination)** | با تکرار حذف ضعیف‌ترین ویژگی‌ها تا رسیدن به بهترین مجموعه |

---

## 💡 ۳. مثال مفهومی ساده

فرض کن ۵ ویژگی داری:
`A, B, C, D, E`

مدل را با هر ترکیب از این ویژگی‌ها اجرا می‌کنی و دقتش را می‌سنجی:

* A → دقت 60%
* A + B → دقت 70%
* A + B + C → دقت 78%
* A + B + C + D → دقت 77%
* A + B + C + D + E → دقت 74%

📊 نتیجه: بهترین ترکیب ویژگی‌ها = A, B, C

---

## 💻 ۴. پیاده‌سازی روش RFE در پایتون

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

# داده‌ی آماده
X, y = load_breast_cancer(return_X_y=True)
model = LogisticRegression(max_iter=2000)

# انتخاب 5 ویژگی برتر
rfe = RFE(model, n_features_to_select=5)
rfe.fit(X, y)

print("Selected Features:")
print(rfe.support_)
print("Feature Ranking:")
print(rfe.ranking_)
```

📊 خروجی نشان می‌دهد کدام ویژگی‌ها در مدل نهایی باقی مانده‌اند (`support=True`).

---

## 🔁 ۵. افزایشی (Forward Selection)

در این روش از هیچ ویژگی شروع می‌کنی و در هر گام ویژگی‌ای را اضافه می‌کنی که **بیشترین بهبود دقت** را ایجاد کند.

📘 شبه‌کد ساده:



```python
from sklearn.linear_model import LogisticRegression

# Simplest data: 3 samples, 2 features
X = [[1, 0], [2, 0], [3, 1]]  # Second feature (0,0,1) is almost useless
y = [0, 0, 1]

model = LogisticRegression()

# Step 1: With all features
model.fit(X, y)
score_all = model.score(X, y)
print(f"With all features: Accuracy = {score_all}")

# Step 2: Only first feature
X1 = [[x[0]] for x in X]
model.fit(X1, y)
score_1 = model.score(X1, y)
print(f"Only first feature: Accuracy = {score_1}")

# Step 3: Only second feature
X2 = [[x[1]] for x in X]
model.fit(X2, y)
score_2 = model.score(X2, y)
print(f"Only second feature: Accuracy = {score_2}")

# Result: Second feature is the weakest
print("\n→ Second feature gets removed because it doesn't improve accuracy")
```

🔹 مزیت: سریع‌تر از تست همه‌ی ترکیب‌ها

🔹 عیب: ممکن است در ابتدای انتخاب، انتخاب اشتباه اثر زنجیره‌ای داشته باشد

---

## 🔄 ۶. حذفی (Backward Elimination)

در مقابل، از همه‌ی ویژگی‌ها شروع می‌کنی و در هر مرحله ضعیف‌ترین ویژگی را حذف می‌کنی.

📘 شبه‌کد:

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

X = [[1,2],[2,3],[3,4],[4,5],[5,6]]  # 2 features only
y = [0,0,1,1,1]

model = LogisticRegression()

# Try with both features
score_all = cross_val_score(model, X, y, cv=2).mean()

# Try with just first feature
X1 = [[x[0]] for x in X]
score_1 = cross_val_score(model, X1, y, cv=2).mean()

# Try with just second feature  
X2 = [[x[1]] for x in X]
score_2 = cross_val_score(model, X2, y, cv=2).mean()

# Remove worst feature
if score_1 >= score_all:
    print(f"Keep only feature 0, Score: {score_1:.2f}")
elif score_2 >= score_all:
    print(f"Keep only feature 1, Score: {score_2:.2f}")
else:
    print(f"Keep both, Score: {score_all:.2f}")
```

🔹 مناسب وقتی ویژگی زیاد داری ولی منابع محاسباتی کافی داری

🔹 اما زمان‌بر است


---

## ⚖️ ۷. مقایسه‌ی روش‌های Wrapper

| ویژگی      | Forward                | Backward        | RFE                  |
| :--------- | :--------------------- | :-------------- | :------------------- |
| شروع       | بدون ویژگی             | با همه ویژگی‌ها | همه ویژگی‌ها         |
| حذف/افزودن | افزودن                 | حذف             | حذف بازگشتی          |
| سرعت       | متوسط                  | کند             | متوسط تا کند         |
| دقت        | بالا                   | بالا            | بسیار بالا           |
| مناسب برای | داده‌های کوچک تا متوسط | داده‌های متوسط  | داده‌های حساس و دقیق |

---

## 🧠 ۸. تمرین عملی

۱️⃣ یک دیتاست با حداقل ۱۰ ویژگی انتخاب کن.
۲️⃣ با `RFE` فقط ۵ ویژگی برتر را انتخاب کن.
۳️⃣ همان مدل را با تمام ویژگی‌ها آموزش بده.
۴️⃣ دقت دو مدل را مقایسه کن — تفاوت را بنویس.

---

## 🎨 تصویر پیشنهادی

> سه فلش افقی رنگی:
>
> * آبی (Forward): در حال اضافه شدن ویژگی‌ها
> * نارنجی (Backward): در حال حذف ویژگی‌ها
> * سبز (RFE): در حال حذف تدریجی و ارزیابی مدل

---

## ❓ پرسش چهارگزینه‌ای

در روش RFE، ویژگی‌ها چگونه انتخاب می‌شوند؟

A) به‌صورت تصادفی

B) با حذف تدریجی ویژگی‌های کم‌اهمیت ✅

C) با محاسبه همبستگی

D) بدون نیاز به مدل

---

