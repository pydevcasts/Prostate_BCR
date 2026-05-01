## 📖 فصل ۹: انیمیشن‌سازی در Matplotlib

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. معرفی

Matplotlib فقط برای نمودارهای ثابت نیست، بلکه می‌توان با کمک ماژول `FuncAnimation` از کتابخانه‌ی `matplotlib.animation` نمودارهای پویا و انیمیشنی ساخت.

این قابلیت برای:

* شبیه‌سازی پدیده‌ها (مثلاً حرکت ذرات)
* آموزش مفاهیم ریاضی (مثل حرکت سینوس)
* تحلیل داده در طول زمان

بسیار مفید است.

---

### 🔹 ۲. انیمیشن ساده (حرکت سینوس)

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

x = np.linspace(0, 2*np.pi, 200) #3.14
y = np.sin(x)

fig, ax = plt.subplots()
line, = ax.plot(x, y)

def update(frame):
    line.set_ydata(np.sin(x + frame/10))  # تغییر تابع در طول زمان
    return line,

ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
plt.show()
```
- `set_ydata()` این متد (تابع) مقدار داده‌های محور y را برای خط (line) به‌روزرسانی می‌کند. به عبارت دیگر، موقعیت.
- `np.sin(x + frame/10)`باعث می‌شود موج سینوسی جا‌به‌جا شود و انیمیشن ایجاد گردد.
- `frames=100` تعداد فریم‌های انیمیشن را تعیین می‌کند؛ یعنی تابع `update` دقیقاً ۱۰۰ بار اجرا می‌شود.  
- `interval=50` فاصله زمانی بین نمایش هر فریم را مشخص می‌کند؛ یعنی هر فریم ۵۰ میلی‌ثانیه روی صفحه می‌ماند.

📌 این کد یک موج سینوسی متحرک ایجاد می‌کند.

---

### 🔹 ۳. انیمیشن Scatter (حرکت نقاط تصادفی)

```python
fig, ax = plt.subplots()
sc = ax.scatter([], [])

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

def init():
    sc.set_offsets([])
    return sc,

def update(frame):
    data = np.random.rand(10, 2)
    sc.set_offsets(data)
    return sc,

ani = FuncAnimation(fig, update, frames=50, interval=200, init_func=init, blit=True)
plt.show()
```

📌 در هر فریم ۱۰ نقطه تصادفی در محدوده (۰،۱) نمایش داده می‌شود.

---

### 🔹 ۴. ذخیره انیمیشن به فایل

می‌توان انیمیشن‌ها را به صورت **MP4** یا **GIF** ذخیره کرد.

```python
ani.save("sine_wave_animation.gif", writer="pillow")
```

📌 این دستور انیمیشن موج سینوسی را به فایل GIF ذخیره می‌کند.

---

### 🔹 نتیجه‌گیری فصل ۹

* با `FuncAnimation` می‌توان نمودارها را پویا کرد.
* انیمیشن‌ها برای آموزش، شبیه‌سازی و ارائه علمی بسیار مفیدند.
* امکان ذخیره خروجی به GIF یا MP4 وجود دارد.

