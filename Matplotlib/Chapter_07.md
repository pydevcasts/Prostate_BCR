## 📖 فصل ۷: اضافه کردن متن و Annotation در نمودارها

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. اضافه کردن متن ساده با `plt.text()`

با این دستور می‌توان متن دلخواه را در یک نقطه از نمودار نوشت.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.text(2, 0.5, "Local Max", fontsize=12, color="red")  # متن در نقطه مشخص
plt.title("Simple Text Example")
plt.show()
```

📌 در این مثال متن "Local Max" روی نمودار در مختصات (۲، ۰.۵) نوشته شده است.

---

### 🔹 ۲. استفاده از `plt.annotate()` برای فلش و توضیح

Annotation پیشرفته‌تر است و می‌توان متن را همراه فلش به یک نقطه متصل کرد.

```python
plt.plot(x, y)
plt.annotate("Peak", xy=(1.57, 1), xytext=(3, 1.2),
             arrowprops=dict(facecolor="black", arrowstyle="->"))
plt.title("Annotation Example")
plt.show()
```

📌 اینجا نقطه‌ی بیشینه تابع سینوس با فلش مشخص شده است.

---

### 🔹 ۳. Highlight کردن یک ناحیه با `axhspan` و `axvspan`

برای نشان دادن بازه‌های مهم از Highlight استفاده می‌کنیم.

```python
plt.plot(x, y)
plt.axhspan(-0.5, 0.5, color="yellow", alpha=0.3)  # ناحیه افقی
plt.axvspan(2, 4, color="lightblue", alpha=0.3)    # ناحیه عمودی
plt.title("Highlight Example")
plt.show()
```

📌 این روش در تحلیل داده‌ها (مثلاً نشان دادن بازه‌های بحرانی) بسیار کاربردی است.

---

### 🔹 ۴. اضافه کردن متن به عنوان Label داده‌ها

گاهی می‌خواهیم مقدار هر نقطه در Scatter Plot نمایش داده شود.

```python
x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 8, 7]

plt.scatter(x, y, color="green")

for i in range(len(x)):
    plt.text(x[i]+0.1, y[i]+0.1, f"({x[i]},{y[i]})", fontsize=9)

plt.title("Scatter Plot with Labels")
plt.show()
```

📌 این روش برای تحلیل داده‌های کوچک یا گزارش‌های آموزشی بسیار کاربردی است.

---

### 🔹 نتیجه‌گیری فصل ۷

* `plt.text()` برای متن ساده.
* `plt.annotate()` برای متن + فلش حرفه‌ای.
* `axhspan` و `axvspan` برای Highlight کردن بازه‌ها.
* اضافه کردن متن به نقاط Scatter برای نمایش دقیق داده‌ها.

