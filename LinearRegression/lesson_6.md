# 📘 فصل ۴: بهبود مدل با رگرسیون منظم‌شده

### ✅ **صفحه ۶– مقابله با Overfitting، هم‌خطی و ویژگی‌های بی‌اثر**

---

## 🎯 چرا باید مدل خود را بهبود دهیم؟

در فصل گذشته، مدل ساده‌ی ما عملکرد خوبی داشت.
اما در داده‌های واقعی معمولاً با مشکلات زیر مواجهیم:

| مشکل                              | تأثیر منفی                            |
| -------------------------------   | ------------------------------------   |
| **ویژگی‌های زیاد و بی‌ربط**        | کاهش دقت مدل                        |
| **هم‌خطی بین ویژگی‌ها**            | ناپایداری ضرایب و تفسیرپذیری دشوار |
| **بیش برازش روی داده آموزشی**  | عملکرد ضعیف روی داده تست            |

برای حل این مشکلات از **مدل‌های Regularized** استفاده می‌کنیم.

---

## 🔧 ۱. Regularization چیست؟

Regularization یعنی **تنبیه کردن ضرایب بزرگ**
هدف: جلوگیری از پیچیده شدن بیش از حد مدل و کاهش نویز.

---

## 📦 ۲. معرفی دو روش رایج Regularization:

| مدل       | نام کامل          | ویژگی اصلی                                                    |
| --------- | ----------------- | ------------------------------------------------------------- |
| **Ridge** | L2 Regularization | کاهش اندازه ضرایب (ولی نگه‌داشتن همه آن‌ها)                   |
| **Lasso** | L1 Regularization | می‌تواند ضرایب بی‌اثر را **صفر کند** (یعنی ویژگی حذف می‌شود!) |

---

## 📐 ۳. فرمول رگرسیون Ridge و Lasso

هر دو مدل Ridge و Lasso، تابع هزینه (Loss Function) خود را با اضافه کردن یک **جریمه (Penalty)** به **مجموع مربعات باقیمانده‌ها (RSS)** تعریف می‌کنند.

- این بخش نشان‌دهنده خطای مدل در پیش‌بینی داده‌های آموزشی است.
    *   $$ RSS = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$
    *   $n$: تعداد نمونه‌های داده آموزشی.
    *   $y_i$: مقدار واقعی (هدف) برای نمونه $i$-ام.
    *   $\hat{y}_i$: مقدار پیش‌بینی شده توسط مدل برای نمونه $i$-ام.
    *   $(y_i - \hat{y}_i)$: **باقیمانده (Residual)** یا خطای پیش‌بینی برای نمونه $i$-ام.
 - هرچه RSS کمتر باشد، مدل برازش (fit) بهتری روی داده‌های آموزشی دارد.

*   **جریمه (Penalty Term):** این بخش، مدل را از پیچیدگی بیش از حد بازمی‌دارد.

### ✅ Ridge:

$$
\text{Loss} = RSS + \alpha \sum_{j=1}^{p} \beta_j^2
$$

### ✅ Lasso:

$$
\text{Loss} = RSS + \alpha \sum_{j=1}^{p} |\beta_j|
$$

* الفا پارامتر تنظیم‌کننده شدت تنبیه
* هر چه α بزرگ‌تر باشد → ضرایب بیشتر کوچک می‌شوند

---

## 💻 ۴. پیاده‌سازی Ridge Regression

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Initialize Ridge model with alpha (regularization strength)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Predict
ridge_pred = ridge_model.predict(X_test)

# Evaluate
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

print(f"Ridge - MSE: {ridge_mse:.3f}, R²: {ridge_r2:.3f}")
```

---

## 💡 نکته مهم:

اگر بین ویژگی‌ها **هم‌خطی (Multicollinearity)** باشد،
مدل Ridge بهتر از LinearRegression معمولی عمل می‌کند چون ضرایب را هموار می‌کند.

---

## 🧬 ۵. پیاده‌سازی Lasso Regression

```python
from sklearn.linear_model import Lasso

# Initialize Lasso model
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

# Predict
lasso_pred = lasso_model.predict(X_test)

# Evaluate
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

print(f"Lasso - MSE: {lasso_mse:.3f}, R²: {lasso_r2:.3f}")
```

---

## 🔍 ۶. تحلیل تفاوت بین Lasso و Ridge

| جنبه                             | Ridge                 | Lasso                       |
| -------------------------------- | --------------------- | --------------------------- |
| تنبیه بر اساس                    | مربع ضرایب (L2)       | قدر مطلق ضرایب (L1)         |
| ضرایب نزدیک صفر                  | بله، ولی صفر نمی‌شوند | بله، بعضی ضرایب صفر می‌شوند |
| انتخاب ویژگی (Feature Selection) | ❌ ندارد               | ✅ دارد                      |
| عملکرد در هم‌خطی                 | بسیار مؤثر            | ممکن است ضرایب را صفر کند   |

---

## 📌 بهترین زمان استفاده از این مدل‌ها:

* **Ridge** → وقتی تمام ویژگی‌ها مفید هستند اما هم‌خطی وجود دارد
* **Lasso** → وقتی فقط برخی ویژگی‌ها مفید هستند (ویژگی‌زدایی خودکار!)

---

