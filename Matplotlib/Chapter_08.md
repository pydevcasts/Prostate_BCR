
## 📖 فصل ۸: تصویرسازی سه‌بعدی

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. ساخت نمودار سه‌بعدی ساده (3D Line Plot)

برای کار با نمودارهای سه‌بعدی باید از **Axes3D** استفاده کنیم.

```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

t = np.linspace(0, 20, 100)
x = np.sin(t)
y = np.cos(t)
z = t

ax.plot3D(x, y, z, color="blue")
ax.set_title("3D Line Plot")
plt.show()
```

📌 این کد یک خط سه‌بعدی مارپیچی را نمایش می‌دهد.

---

### 🔹 ۲. Scatter سه‌بعدی (3D Scatter Plot)

```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.random.rand(50)
y = np.random.rand(50)
z = np.random.rand(50)
colors = np.random.rand(50)

ax.scatter(x, y, z, c=colors, cmap="viridis", s=60)
ax.set_title("3D Scatter Plot")
plt.show()
```

📌 نقاط سه‌بعدی با رنگ‌بندی بر اساس مقدار `colors` نمایش داده می‌شوند.

---

### 🔹 ۳. سطح سه‌بعدی (Surface Plot)

```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))

surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", edgecolor="none")
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.set_title("3D Surface Plot")
plt.show()
```

📌 این نمودار سطحی سه‌بعدی از تابع سینوسی شعاعی را نشان می‌دهد.

---

### 🔹 ۴. Wireframe Plot (قاب سه‌بعدی)

```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(X, Y, Z, color="black")
ax.set_title("3D Wireframe Plot")
plt.show()
```

📌 مشابه Surface Plot است اما فقط خطوط شبکه‌ای نمایش داده می‌شوند.

---

### 🔹 نتیجه‌گیری فصل ۸

* `plot3D` برای خطوط سه‌بعدی.
* `scatter` برای نمایش نقاط سه‌بعدی.
* `surface` برای نمایش سطوح ریاضی و داده‌های شبکه‌ای.
* `wireframe` برای نمایش ساختار شبکه‌ای به شکل ساده‌تر.

