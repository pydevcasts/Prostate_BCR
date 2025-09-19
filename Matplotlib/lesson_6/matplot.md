

## 📊 آموزش تصویری با matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np
```

### 📈 نمودار خطی با رنگ قرمز و نشانگر دایره

```python
ypoint = np.array([2,4,1,5])
plt.figure()
plt.plot(ypoint, "r", marker="o")
plt.show()
```

### 📈 رسم یک نمودار خطی ساده

```python
plt.plot([1, 3, 2], [2, 4, 2])
plt.show()
```

### 🥧 نمودار دایره‌ای (Pie Chart)

```python
x = ["red", "green", "yellow", "blue"]
y = [5, 6, 7, 8]
plt.pie(y, labels=x, autopct="%1.1f%%", colors=["red", "green", "yellow", "blue"])
plt.show()
```

### 📊 نمودار میله‌ای با رنگ سبز

```python
x = ["red", "green", "yellow", "blue"]
y = [5, 6, 7, 8]
plt.bar(x, y, color="green")
plt.show()
```

### 📉 هیستوگرام از داده‌های تصادفی

```python
x = np.random.rand(1000, 1)
plt.hist(x, bins=20)
plt.show()
```

### 🌡️ نمایش یک ماتریس به‌صورت نقشه حرارتی

```python
a = np.random.rand(64, 64)
plt.imshow(a, cmap="hot")
plt.colorbar()
plt.show()
```

### 📊 نمودار ترکیبی با تنظیمات مختلف

```python
x = np.array([10, 4, 7, 1])
y = [2, 8, 3, 9]
plt.plot(x, marker="^", ms=25, mec="g", mfc="y")
plt.plot(y, "o:r")
plt.legend(["Dataset1", "Dataset2"])
plt.show()
```

### 📝 افزودن عنوان و شبکه‌بندی به نمودار

```python
font1 = {"family": "serif", "color": "blue", "size": 20}
plt.grid(axis="y", color="green", linestyle="--", linewidth=1.5)
plt.title("title", fontdict=font1, loc='center')
```

