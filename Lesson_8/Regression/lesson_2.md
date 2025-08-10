
# 📘 فصل ۲: آموزش مدل ساده با کد و تحلیل واقعی

### ✅ **صفحه ۲ – ارزیابی کیفیت مدل، بررسی خطاها و قدرت پیش‌بینی**

---

## 🎯 هدف این صفحه:

اکنون که مدل رگرسیون خطی را ساختیم، سؤال اصلی این است:

> «آیا این مدل به اندازه کافی خوب هست که به آن در پروژه واقعی تکیه کنیم؟»

در این صفحه، با ابزارهایی مثل **نمودار خطاها، شاخص‌های آماری، و تحلیل بصری** به این پرسش پاسخ خواهیم داد.

---

## 🔍 ۱. بررسی تصویری تطابق مدل با داده‌ها

یکی از ساده‌ترین روش‌ها برای درک کیفیت یک مدل رگرسیون، **رسم نمودار مقایسه‌ای بین داده واقعی و خط برازش‌شده** است.

🔵 داده‌ها (واقعی)

🔴 خط رگرسیون (پیش‌بینی‌شده)

فرض کنید ما داده‌های مربوط به زمان مطالعه و نمرات دانش‌آموزان را داریم و قصد داریم کیفیت مدل رگرسیون خطی خود را ارزیابی کنیم.

### ۱. بررسی تصویری تطابق مدل با داده‌ها
ابتدا، ما می‌خواهیم نمودار مقایسه‌ای بین داده‌های واقعی و خط رگرسیون را رسم کنیم.

#### کد نمونه:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample data: Hours of study (x) and corresponding scores (y)
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Reshaping x for the model (1D to 2D)
y = np.array([55, 60, 65, 70, 75])  # Scores

# Create a linear regression model and fit it to the data
model = LinearRegression()
model.fit(x, y)  # Fitting the model to the training data
y_pred = model.predict(x)  # Predicting scores based on the model

# Predicting the score for 6 hours of study
hours_to_predict = np.array([[6]])  # Reshaping for a single prediction
predicted_score = model.predict(hours_to_predict)  # Predicting score

# Output the predicted score
print(f"Predicted score for {hours_to_predict[0][0]} hours of study: {predicted_score[0]:.2f}")

# Plotting the actual data and the regression line
plt.scatter(x, y, color='blue', label='Actual Data')  # Scatter plot for actual scores
plt.plot(x, y_pred, color='red', label='Regression Line')  # Line for predicted scores
plt.scatter(hours_to_predict, predicted_score, color='green', label='Predicted Score (6 hours)')  # Point for predicted score
plt.xlabel('Study Hours')  # Label for x-axis
plt.ylabel('Score')  # Label for y-axis
plt.title('Comparison of Actual Data and Regression Line')  # Title of the plot
plt.legend()  # Show legend
plt.grid(True)  # Enable grid for better readability
plt.show()  # Display the plot
```


## 📉 ۲. رسم نمودار خطا (Residual Plot)

### ❗ چرا مهم است؟

اگر مدل خوبی داشته باشیم، **خطاها (residuals)** باید به‌صورت تصادفی و بدون الگو پخش شده باشند.

### ✅ پیاده‌سازی:

```python
# Calculate residuals
residuals = y - y_pred

# Plot residuals
plt.scatter(x, residuals, color='purple')
plt.axhline(y=0, color='gray', linestyle='--')
plt.xlabel('x')
plt.ylabel('Residual (y - y_pred)')
plt.title('Residual Plot')
plt.grid(True)
plt.show()
```

📌 اگر نقاط اطراف محور افقی به‌صورت تصادفی و متقارن پخش شده باشند، یعنی مدل خوب برازش شده.
اما اگر الگویی خاص یا روند افزایشی/کاهشی دیده شود، ممکن است مدل مناسب نباشد (مثلاً مدل خطی برای داده غیرخطی استفاده شده است).

---

## 📊 ۳. معیارهای ارزیابی مدل: R² و MSE

### ✅ R² – Coefficient of Determination:

 مقدار بین ۰ تا ۱ است.
هر چه به ۱ نزدیک‌تر باشد، مدل بهتر داده‌ها را توضیح می‌دهد.

---

### ✅ MSE – Mean Squared Error:

یک معیار عددی برای میانگین توان دوم خطاها:

$$
MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2
$$

📌 هر چه مقدار MSE کوچک‌تر باشد، مدل پیش‌بینی بهتری انجام داده است.

### 🔧 پیاده‌سازی:

```python
from sklearn.metrics import mean_squared_error, r2_score

# Calculate R² and Mean Squared Error (MSE)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

# Output R² and MSE
print(f"R-squared (R²): {r2:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")


# Get the bias (intercept) and slope of the model
bias = model.intercept_  # Intercept (bias)
slope = model.coef_[0]  # Slope (coefficient)

# Output Bias and Slope
print(f"Bias (Intercept): {bias:.2f}")
print(f"Slope (Coefficient): {slope:.2f}")

# Display the regression equation
print(f"\nRegression Equation: y = {bias:.2f} + {slope:.2f} * x")

```

---

## 📋 ۴. تفسیر خروجی‌ها در دنیای واقعی

فرض کنیم خروجی ما این باشد:

```python
R-squared (R²): 1.00
Mean Squared Error (MSE): 0.00
Bias (Intercept): 50.00
Slope (Coefficient): 5.00

Regression Equation: y = 50.00 + 5.00 * x
```

معنایش چیست؟

* MSE = 0.0 یعنی میانگین انحراف پیش‌بینی‌ها از واقعیت فقط 0.0 واحد است – خیلی خوب!
* R² = 1 یعنی 100% از واریانس داده‌ها توسط مدل قابل توضیح است.

---

## 📌 نکته مهم:

حتی اگر مدل بسیار دقیق به‌نظر برسد، باید در نظر بگیریم که:

* آیا داده‌ها **نماینده کافی** از وضعیت واقعی هستند؟
* آیا تعداد نقاط کافی است؟ (۵ داده برای پروژه واقعی بسیار کم است!)
* آیا مدل **در داده‌های جدید** نیز خوب عمل می‌کند؟ (اینجا نیاز به تست روی داده unseen داریم، که در فصل پروژه واقعی بررسی خواهیم کرد)

---

## 🧭 ۵. آیا مدل بیش‌برازش کرده است؟ (Overfitting)

در رگرسیون خطی ساده، احتمال Overfitting بسیار پایین است
اما اگر تعداد داده‌ها کم و مدل بر همه نقاط دقیقاً منطبق باشد (مثلاً R² = 1)، مشکوک می‌شویم!

> **مثال کلاسیک Overfitting:** مدل شما دقیقاً داده‌های موجود را حفظ کرده، ولی نمی‌تواند داده‌های جدید را پیش‌بینی کند.
---
