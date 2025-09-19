## 📖 فصل ۶: استایل‌ها و تنظیمات حرفه‌ای

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. استفاده از Style Sheets آماده

Matplotlib چندین استایل آماده دارد که می‌توانند ظاهر نمودارها را تغییر دهند.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.style.use("ggplot")  # تغییر استایل به ggplot
plt.plot(x, y)
plt.title("Sine Function with ggplot Style")
plt.show()
```

📌 استایل‌ها شامل: `"seaborn"`, `"ggplot"`, `"fivethirtyeight"`, `"dark_background"`, `"classic"` و ... هستند.

---

### 🔹 ۲. تغییر رنگ‌بندی‌ها (Colormaps)

```python
np.random.seed(42)
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)

plt.scatter(x, y, c=colors, cmap="viridis", s=100)
plt.colorbar()  # اضافه کردن نوار رنگ
plt.title("Scatter Plot with Colormap")
plt.show()
```

📌 با `cmap` می‌توان رنگ‌های مختلف مثل `"viridis"`, `"plasma"`, `"coolwarm"` و ... را امتحان کرد.

---

### 🔹 ۳. تغییر اندازه و رزولوشن تصویر

```python
plt.figure(figsize=(12,6), dpi=100)  # dpi برای رزولوشن
plt.plot(x, y, color="red")
plt.title("High Resolution Plot")
plt.show()
```

📌 با `dpi` می‌توان کیفیت نمودار را برای چاپ یا ارائه بالا برد.

---

### 🔹 ۴. تغییر ظاهر خطوط و نشانگرها

```python
plt.plot(x, y, color="blue", linewidth=3, linestyle="--", marker="o", markersize=8, markerfacecolor="yellow")
plt.title("Customized Line and Markers")
plt.show()
```

📌 می‌توان رنگ داخلی نشانگر (`markerfacecolor`) و ضخامت خط (`linewidth`) را تغییر داد.

---

### 🔹 ۵. ذخیره نمودارها در فایل

```python
plt.plot(x, y, color="green")
plt.title("Save Figure Example")
plt.savefig("plot_example.png", dpi=300, bbox_inches="tight")
plt.show()
```

📌 خروجی در قالب PNG با کیفیت بالا ذخیره می‌شود. همچنین می‌توان فرمت‌های دیگر مثل **PDF**, **SVG**, **JPG** را هم ذخیره کرد.

---

### 🔹 نتیجه‌گیری فصل ۶

* با Style Sheets می‌توان ظاهر کلی نمودار را تغییر داد.
* Colormap برای رنگ‌بندی داده‌های عددی بسیار مفید است.
* می‌توان کیفیت نمودار را با DPI افزایش داد.
* نمودارها به راحتی ذخیره می‌شوند و برای گزارش‌های علمی یا مقالات آماده‌اند.
