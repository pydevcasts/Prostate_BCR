عالی! 🎯
اکنون ادامه می‌دهیم با **صفحه ۲ از فصل ۳** — اینجاست که مدل واقعی رگرسیون چندمتغیره را روی داده‌ای صنعتی آموزش می‌دهیم، آن را ارزیابی می‌کنیم و نتایج را **بصری و عددی تحلیل** می‌کنیم.

---

# 📘 فصل ۳: پروژه واقعی – پیش‌بینی قیمت خانه

### ✅ **صفحه ۵ – آموزش مدل، ارزیابی و تحلیل نتایج**

---

## 🧠 ۱. آموزش مدل با Scikit-learn

اکنون که داده‌ها را آماده کرده‌ایم، وقت آن است که مدل رگرسیون را روی **داده‌های آموزشی (train)** آموزش دهیم.

```python
from sklearn.linear_model import LinearRegression

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Print coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
```

### 📌 تفسیر ضرایب:

* هر عدد در `model.coef_` نشان می‌دهد چقدر تغییر در ویژگی مربوطه، روی قیمت خانه تأثیر دارد
* `model.intercept_` یعنی مقدار پایه قیمت خانه، وقتی همه ویژگی‌ها صفر باشند (در تئوری)

---

## 🧪 ۲. پیش‌بینی قیمت روی داده تست (Test Set)

حالا مدل را روی داده‌ای که **در آموزش استفاده نشده** تست می‌کنیم:

```python
# Predict on test set
y_pred = model.predict(X_test)
```

---

## 📊 ۳. ارزیابی دقت مدل

### ✅ محاسبه MSE و R²:

```python
from sklearn.metrics import mean_squared_error, r2_score

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.3f}")
print(f"R-squared: {r2:.3f}")
```

### 🎯 تفسیر:

| شاخص           | توضیح                                               |
| -------------- | --------------------------------------------------- |
| **MSE پایین**  | پیش‌بینی‌ها به مقادیر واقعی نزدیک هستند             |
| **R² نزدیک ۱** | مدل به‌خوبی توانسته است تغییرات y را با X توضیح دهد |

---

## 📈 ۴. نمودار مقایسه قیمت واقعی و پیش‌بینی‌شده

این نمودار یکی از مهم‌ترین ابزارهای بصری در پروژه‌های واقعی است.
در صورت ایده‌آل، نقاط باید نزدیک به خط قطر قرار بگیرند.

```python
import matplotlib.pyplot as plt

# Plot predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         '--', color='gray')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()
```

---

## 📌 اگر نقاط خیلی پراکنده باشند...

📉 یعنی مدل هنوز خیلی خوب برازش نکرده و ممکن است:

* ویژگی‌های بهتری باید انتخاب شود
* روابط غیرخطی در داده وجود داشته باشد
* برخی داده‌ها پرت باشند
* نیاز به regularization (مثلاً Ridge/Lasso) باشد → در فصل بعدی

---

## 🔍 ۵. بررسی ویژگی‌های تأثیرگذار در مدل

بیایید ضرایب را با نام ویژگی‌ها همراه کنیم:

```python
# Combine feature names with their coefficients
for name, coef in zip(selected_features, model.coef_):
    print(f"{name}: {coef:.3f}")
```

> این خروجی به ما می‌گوید **کدام ویژگی‌ها بیشترین تأثیر مثبت یا منفی را دارند.**

---

## 🧭 ۶. آیا مدل برای استفاده در پروژه واقعی آماده است؟

✅ اگر MSE پایین و R² بالاتر از \~0.6 یا 0.7 باشد (در داده‌های دنیای واقعی)، معمولاً مدل می‌تواند در شرایط واقعی مورد استفاده قرار گیرد.
🔎 در ادامه، می‌توان مدل را با روش‌های پیشرفته‌تر بهبود داد.

---




