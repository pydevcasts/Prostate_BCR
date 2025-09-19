## 📖 فصل ۵: Subplots در Matplotlib

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. چند نمودار ساده در یک تصویر

گاهی لازم است چند نمودار مختلف را کنار هم نمایش دهیم.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.subplot(1, 2, 1)  # (rows, cols, index)
plt.plot(x, y1, color="blue")
plt.title("Sine Function")

plt.subplot(1, 2, 2)
plt.plot(x, y2, color="red")
plt.title("Cosine Function")

plt.show()
```

📌 این کد دو نمودار (sin و cos) را در یک ردیف کنار هم رسم می‌کند.

---

### 🔹 ۲. Subplots در چند ردیف و ستون

```python
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = np.exp(-x)

plt.figure(figsize=(10,8))

plt.subplot(2, 2, 1)
plt.plot(x, y1)
plt.title("Sine")

plt.subplot(2, 2, 2)
plt.plot(x, y2)
plt.title("Cosine")

plt.subplot(2, 2, 3)
plt.plot(x, y3)
plt.title("Tangent")

plt.subplot(2, 2, 4)
plt.plot(x, y4)
plt.title("Exponential Decay")

plt.tight_layout()
plt.show()
```

📌 این کد ۴ نمودار را در قالب یک **ماتریس ۲×۲** نمایش می‌دهد.

---

### 🔹 ۳. استفاده از `plt.subplots()` برای کنترل بهتر

```python
fig, axs = plt.subplots(2, 2, figsize=(10,8))

axs[0, 0].plot(x, y1, color="blue")
axs[0, 0].set_title("Sine")

axs[0, 1].plot(x, y2, color="red")
axs[0, 1].set_title("Cosine")

axs[1, 0].plot(x, y3, color="green")
axs[1, 0].set_title("Tangent")

axs[1, 1].plot(x, y4, color="purple")
axs[1, 1].set_title("Exponential Decay")

plt.tight_layout()
plt.show()
```

📌 این روش حرفه‌ای‌تر است چون به شما اجازه می‌دهد به هر subplot به صورت جداگانه دسترسی داشته باشید و تنظیمات بیشتری روی آن انجام دهید.

---

### 🔹 ۴. اشتراک‌گذاری محور‌ها (Shared Axis)

```python
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8,6))

axs[0].plot(x, y1, label="Sine")
axs[0].legend()

axs[1].plot(x, y2, label="Cosine", color="orange")
axs[1].legend()

plt.show()
```

📌 در اینجا محور **X** بین دو نمودار مشترک است، که مقایسه آن‌ها را آسان‌تر می‌کند.

---

### 🔹 نتیجه‌گیری فصل ۵

* `subplot()` → روش ساده برای چند نمودار.
* `subplots()` → کنترل کامل و حرفه‌ای‌تر.
* می‌توان محورهای مشترک ایجاد کرد و از `tight_layout()` برای تنظیم فاصله‌ها استفاده کرد.
