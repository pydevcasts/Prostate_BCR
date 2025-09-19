## 📖 فصل ۳: سفارشی‌سازی نمودارها

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. تغییر رنگ، خط و Marker

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 5, 3, 7, 9]

plt.plot(x, y, color="purple", linestyle="--", marker="o", linewidth=2, markersize=8)
plt.title("Customized Line Plot")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.show()
```

📌 می‌توانیم با تغییر `color`، `linestyle`، `marker` و `linewidth` نمودار را سفارشی کنیم.

---

### 🔹 ۲. اضافه کردن Legend (راهنما)

```python
x = [1, 2, 3, 4, 5]
y1 = [2, 4, 6, 8, 10]
y2 = [1, 3, 5, 7, 9]

plt.plot(x, y1, label="Line 1", color="blue")
plt.plot(x, y2, label="Line 2", color="red")

plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Line Plot with Legend")
plt.legend()
plt.show()
```

📌 `legend()` برای نمایش راهنما استفاده می‌شود.

---

### 🔹 ۳. اضافه کردن Grid (شبکه)

```python
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

plt.plot(x, y, color="green", marker="o")
plt.title("Line Plot with Grid")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
```

📌 Grid کمک می‌کند داده‌ها بهتر خوانده شوند.

---

### 🔹 ۴. تغییر اندازه نمودار

```python
plt.figure(figsize=(8,5))
plt.plot(x, y, color="orange", marker="s")
plt.title("Resized Plot Example")
plt.show()
```

📌 با `figsize=(عرض, ارتفاع)` می‌توانیم ابعاد نمودار را تغییر دهیم.

---

### 🔹 ۵. تغییر فونت و رنگ عنوان‌ها

```python
plt.plot(x, y, color="black", marker="d")
plt.title("Styled Title", fontsize=16, color="red")
plt.xlabel("X Axis", fontsize=12, color="blue")
plt.ylabel("Y Axis", fontsize=12, color="blue")
plt.show()
```

📌 با `fontsize` و `color` می‌توانیم فونت و رنگ عنوان‌ها را تنظیم کنیم.

---

### 🔹 نتیجه‌گیری فصل ۳

* امکان تغییر رنگ، خط، Marker وجود دارد.
* می‌توان Legend، Grid و اندازه نمودار را تنظیم کرد.
* فونت و رنگ‌ها قابل سفارشی‌سازی هستند.
