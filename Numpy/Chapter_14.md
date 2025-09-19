
# 📖 فصل ۱۴: ترکیب NumPy با Matplotlib برای مصورسازی داده‌ها

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. ترسیم داده‌های خطی

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, color="blue", label="sin(x)")
plt.title("Sine Function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
```

📌 NumPy برای تولید داده‌ها و Matplotlib برای نمایش آن‌ها کنار هم استفاده می‌شوند.

---

### 🔹 ۲. نمودار چند تابع در یک شکل

```python
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin(x)", color="red")
plt.plot(x, y2, label="cos(x)", color="green")
plt.title("Sine and Cosine")
plt.legend()
plt.show()
```

📌 می‌توان چند داده مختلف را در یک نمودار ترسیم کرد.

---

### 🔹 ۳. نمودار پراکندگی (Scatter Plot)

```python
x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x, y, color="purple", marker="o")
plt.title("Scatter Plot")
plt.show()
```

📌 مناسب برای نمایش رابطه بین دو متغیر.

---

### 🔹 ۴. هیستوگرام داده‌ها

```python
data = np.random.normal(0, 1, 1000)

plt.hist(data, bins=30, color="skyblue", edgecolor="black")
plt.title("Histogram")
plt.show()
```

📌 برای نمایش توزیع داده‌ها استفاده می‌شود.

---

### 🔹 ۵. نمودار میله‌ای (Bar Plot)

```python
categories = ["A", "B", "C", "D"]
values = [5, 7, 3, 8]

plt.bar(categories, values, color="orange")
plt.title("Bar Plot Example")
plt.show()
```

📌 نمودار میله‌ای برای مقایسه دسته‌ها کاربرد دارد.

---

### 🔹 ۶. نمودار در چند Subplot

```python
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.subplot(2, 1, 1)
plt.plot(x, y1, color="red")
plt.title("Sine")

plt.subplot(2, 1, 2)
plt.plot(x, y2, color="blue")
plt.title("Cosine")

plt.tight_layout()
plt.show()
```

📌 Subplot برای نمایش چند نمودار در یک شکل استفاده می‌شود.

---

### 🔹 ۷. مثال ترکیبی 🎨

```python
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10,5))

plt.plot(x, y1, label="sin(x)", color="green")
plt.plot(x, y2, label="cos(x)", color="orange")
plt.fill_between(x, y1, y2, alpha=0.3)  # فضای بین دو منحنی

plt.title("Sine vs Cosine")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
```

📌 اینجا علاوه بر ترسیم توابع، فضای بین آن‌ها هم نمایش داده شده است.

---

### 🔹 ۸. تمرین پیشنهادی 🎯

۱. نمودار تابع $y = x^2$ و $y = x^3$ را روی یک نمودار بکشید.
۲. داده‌های تصادفی با توزیع نرمال بسازید و هیستوگرام آن را رسم کنید.
۳. یک نمودار میله‌ای از فروش ۵ محصول فرضی ترسیم کنید.
۴. نمودار سینوس و کسینوس را در دو Subplot جداگانه نمایش دهید.

