## 📖 فصل ۱: مقدمه‌ای بر Matplotlib

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 Matplotlib چیست؟

Matplotlib یک کتابخانه قدرتمند در زبان پایتون برای **ترسیم نمودارها و تصویرسازی داده‌ها** است.
این کتابخانه یکی از پایه‌ای‌ترین ابزارها در **علم داده (Data Science)** و **یادگیری ماشین (Machine Learning)** محسوب می‌شود.

📌 با Matplotlib می‌توانید:

* نمودارهای خطی، ستونی، پراکندگی و آماری رسم کنید.
* داده‌های علمی و پیچیده را به شکل ساده و قابل فهم نمایش دهید.
* نمودارها را سفارشی‌سازی کنید (رنگ، فونت، استایل، عنوان و ...)

---

### 🔹 نصب Matplotlib

برای نصب کافی است دستور زیر را در ترمینال یا Jupyter Notebook اجرا کنید:

```bash
pip install matplotlib
```

---

### 🔹 اولین کد ساده با Matplotlib

```python
import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create line plot
plt.plot(x, y)

# Add labels and title
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Simple Line Plot")

# Show the plot
plt.show()
```

---

### 🔹 توضیح کد:

1. `import matplotlib.pyplot as plt` → فراخوانی ماژول ترسیم.
2. `x` و `y` → داده‌هایی که می‌خواهیم رسم کنیم.
3. `plt.plot(x, y)` → رسم نمودار خطی بین نقاط.
4. `plt.xlabel` و `plt.ylabel` → برچسب برای محورهای افقی و عمودی.
5. `plt.title` → عنوان نمودار.
6. `plt.show()` → نمایش نمودار.

---

### 🔹 خروجی کد:

📊 نمودار خطی ساده‌ای که نشان می‌دهد مقدار y دو برابر مقدار x است.


