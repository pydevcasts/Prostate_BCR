
# 📘 فصل ۴: بهبود مدل با رگرسیون منظم‌شده

### ✅ **صفحه ۷ – مقایسه عملکرد Ridge و Lasso + نتیجه‌گیری نهایی پروژه**

---

## 🎯 هدف این صفحه:

اکنون که مدل‌های Ridge و Lasso را آموزش داده‌ایم، وقت آن است که:

* دقت آن‌ها را مقایسه کنیم
* ضرایب خروجی هر مدل را بررسی کنیم
* نمودار عملکرد آن‌ها را ببینیم
* و تصمیم بگیریم: **کدام مدل را در پروژه نهایی استفاده کنیم؟**

---

## 📊 ۱. مقایسه ضرایب مدل‌ها

بیایید ضرایب مدل‌های Linear, Ridge و Lasso را کنار هم چاپ کنیم:

```python
# Display coefficients from all models
models = {
    "Linear": model,
    "Ridge": ridge_model,
    "Lasso": lasso_model
}

for name, m in models.items():
    print(f"\n{name} Coefficients:")
    for feature, coef in zip(selected_features, m.coef_):
        print(f"{feature}: {coef:.4f}")
```

### 🎯 چه چیزی را بررسی می‌کنیم؟

* کدام مدل ضرایب بسیار بزرگ یا منفی دارد؟
* آیا Lasso بعضی ویژگی‌ها را به صفر رسانده؟ (یعنی آن ویژگی حذف شده است)
* آیا مدل Ridge ضرایب را "صاف‌تر" کرده است؟

---

## 📉 ۲. رسم نمودار Actual vs Predicted برای هر مدل

```python
def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.5, color='darkorange')
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             '--', color='gray')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(title)
    plt.grid(True)
    plt.show()

# Plot for each model
plot_predictions(y_test, model.predict(X_test), "Linear Regression")
plot_predictions(y_test, ridge_model.predict(X_test), "Ridge Regression")
plot_predictions(y_test, lasso_model.predict(X_test), "Lasso Regression")
```

📌 **هرچه نقاط به خط قطر نزدیک‌تر باشند، مدل دقیق‌تر است.**

---

## 🧪 ۳. مقایسه عددی MSE و R² برای هر مدل

```python
for name, m in models.items():
    y_hat = m.predict(X_test)
    mse = mean_squared_error(y_test, y_hat)
    r2 = r2_score(y_test, y_hat)
    print(f"{name} → MSE: {mse:.3f}, R²: {r2:.3f}")
```

### تفسیر نتایج:

| مدل    | MSE ↓                | R² ↑                      | تفسیر                               |
| ------ | -------------------- | ------------------------- | ----------------------------------- |
| Linear | معمولاً دقیق         | حساس به outlier           | پایه اولیه خوب                      |
| Ridge  | دقیق‌تر              | پایداری بالا              | عالی در هم‌خطی                      |
| Lasso  | ممکن است ساده‌تر شود | برخی ویژگی‌ها حذف می‌شوند | مناسب زمانی که تعداد ویژگی زیاد است |

---

## 🤔 ۴. انتخاب مدل نهایی برای پروژه

| سوال عملیاتی                                | پاسخ                                              |
| ------------------------------------------- | ------------------------------------------------- |
| آیا داده‌ها نویزی‌اند؟                      | بله، پس مدل با Regularization بهتر است            |
| آیا بین ویژگی‌ها هم‌خطی وجود دارد؟          | بله، `AveRooms` و `AveOccup` ممکن است مرتبط باشند |
| آیا قصد داریم ویژگی‌های بی‌اثر را حذف کنیم؟ | بله، پس Lasso مفید است                            |

---

## ✅ نتیجه‌گیری:

> اگر هدف ما **پیش‌بینی دقیق و پایداری در داده واقعی** است:
> **Ridge Regression بهترین گزینه ماست.**

> اگر هدف ما **ساده‌سازی مدل و حذف ویژگی‌های کم‌اثر** است:
> **Lasso Regression انتخاب هوشمندانه‌تری است.**

---

## ✅ کتاب تمام شد — اما یادگیری ادامه دارد...

شما اکنون با مفاهیم پایه، پیاده‌سازی کامل، تفسیر خروجی‌ها، بهبود مدل، و کاربرد واقعی رگرسیون خطی کاملاً مسلط هستید.

---

### 🔚 پایان کتاب:

**درک عمیق رگرسیون خطی — راهنمایی کاربردی برای تحلیل‌گران و دانشجویان داده**
✍️ *نویسنده: سیامک عباس‌نژاد (pydevcasts)*

---
