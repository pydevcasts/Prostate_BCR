# 📖 فصل ۱۱: تجسم داده‌ها با Pandas و Matplotlib

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. مقدمه

Pandas به صورت داخلی از **Matplotlib** پشتیبانی می‌کند و می‌توانیم خیلی راحت روی داده‌ها نمودار بکشیم.
با این روش می‌توانیم سریع داده‌ها را بررسی و تحلیل کنیم.

---

### 🔹 ۲. نمودار خطی (Line Plot)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame
data = pd.DataFrame({
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "Sales": [100, 120, 90, 150, 200, 180]
})

# Plot line chart
data.plot(x="Month", y="Sales", kind="line", marker="o")
plt.title("Monthly Sales")
plt.ylabel("Sales")
plt.show()
```

---

### 🔹 ۳. نمودار میله‌ای (Bar Plot)

```python
# Plot bar chart
data.plot(x="Month", y="Sales", kind="bar", color="skyblue")
plt.title("Monthly Sales (Bar)")
plt.ylabel("Sales")
plt.show()
```

---

### 🔹 ۴. نمودار میله‌ای افقی

```python
# Plot horizontal bar chart
data.plot(x="Month", y="Sales", kind="barh", color="orange")
plt.title("Monthly Sales (Horizontal)")
plt.xlabel("Sales")
plt.show()
```

---

### 🔹 ۵. نمودار هیستوگرام (Histogram)

```python
# Histogram example
df = pd.DataFrame({
    "Values": [5, 7, 8, 5, 6, 7, 8, 7, 6, 7, 8, 9, 10, 8, 7, 6]
})

df["Values"].plot(kind="hist", bins=5, color="purple", alpha=0.7)
plt.title("Histogram Example")
plt.show()
```

---

### 🔹 ۶. نمودار جعبه‌ای (Boxplot)

```python
# Boxplot example
df2 = pd.DataFrame({
    "A": [10, 20, 30, 40, 100],
    "B": [15, 25, 35, 45, 55]
})

df2.plot(kind="box")
plt.title("Boxplot Example")
plt.show()
```

---

### 🔹 ۷. نمودار پراکندگی (Scatter Plot)

```python
# Scatter plot example
df3 = pd.DataFrame({
    "x": [5, 7, 8, 7, 6, 9, 5, 6, 7, 8],
    "y": [99, 86, 87, 88, 100, 86, 103, 87, 94, 78]
})

df3.plot(kind="scatter", x="x", y="y", color="red")
plt.title("Scatter Plot Example")
plt.show()
```

---

### 🔹 ۸. تمرین پیشنهادی

۱. یک دیتافریم شامل فروش ۴ محصول مختلف در طول یک سال بسازید.
۲. نمودار خطی برای مقایسه فروش محصولات رسم کنید.
۳. نمودار هیستوگرام برای یکی از محصولات بکشید.
۴. نمودار Boxplot برای مقایسه فروش کل محصولات رسم کنید.

