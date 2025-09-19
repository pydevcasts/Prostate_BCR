
# 📖 فصل ۱۱: توابع آماری و ریاضی در NumPy

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. محاسبات پایه‌ای آماری

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])

print("Sum:", np.sum(arr))       # مجموع
print("Mean:", np.mean(arr))     # میانگین
print("Median:", np.median(arr)) # میانه
print("Std:", np.std(arr))       # انحراف معیار
print("Var:", np.var(arr))       # واریانس
```

📌 این توابع پرکاربردترین ابزارهای آماری NumPy هستند.

---

### 🔹 ۲. محاسبات روی چند بعد

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print("Sum of all:", np.sum(matrix))
print("Sum per row:", np.sum(matrix, axis=1))  # مجموع هر سطر
print("Sum per col:", np.sum(matrix, axis=0))  # مجموع هر ستون
```

📌 پارامتر **axis** مشخص می‌کند محاسبه روی کدام بُعد انجام شود.

---

### 🔹 ۳. حداقل و حداکثر

```python
arr = np.random.randint(1, 100, 10)

print("Array:", arr)
print("Min:", np.min(arr))
print("Max:", np.max(arr))
print("Argmin (index):", np.argmin(arr))
print("Argmax (index):", np.argmax(arr))
```

📌 `argmin` و `argmax` اندیس کوچک‌ترین و بزرگ‌ترین مقدار را می‌دهند.

---

### 🔹 ۴. Percentile و Quantile

```python
arr = np.array([7, 8, 5, 10, 15, 20])

print("25th Percentile:", np.percentile(arr, 25))
print("50th Percentile (Median):", np.percentile(arr, 50))
print("75th Percentile:", np.percentile(arr, 75))

print("Quantile 0.25:", np.quantile(arr, 0.25))
```

📌 پرسن‌تایل‌ها در آمار کاربرد زیادی دارند، مثلاً در تحلیل داده‌های پزشکی یا اقتصادی.

---

### 🔹 ۵. توابع ریاضی پایه‌ای

```python
x = np.linspace(0, 2*np.pi, 5)

print("sin:", np.sin(x))
print("cos:", np.cos(x))
print("tan:", np.tan(x))

print("exp:", np.exp([1, 2, 3]))
print("log:", np.log([1, 2, 3]))
print("sqrt:", np.sqrt([4, 9, 16]))
```

📌 این توابع مخصوص محاسبات علمی و مهندسی هستند.

---

### 🔹 ۶. مثال ترکیبی با مصورسازی 📊

```python
import matplotlib.pyplot as plt

data = np.random.normal(50, 10, 1000)  # داده نرمال با میانگین 50 و انحراف معیار 10

mean = np.mean(data)
std = np.std(data)

plt.hist(data, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
plt.axvline(mean, color="red", linestyle="--", label=f"Mean = {mean:.2f}")
plt.axvline(mean+std, color="green", linestyle="--", label=f"+1 Std = {mean+std:.2f}")
plt.axvline(mean-std, color="green", linestyle="--", label=f"-1 Std = {mean-std:.2f}")

plt.title("Histogram with Mean and Std")
plt.legend()
plt.show()
```

📌 این نمودار به ما نشان می‌دهد داده‌ها چطور حول میانگین توزیع شده‌اند.

---

### 🔹 ۷. تمرین پیشنهادی 🎯

۱. یک آرایه تصادفی از ۱۰۰۰ عدد تولید کنید.
۲. میانگین، میانه، انحراف معیار و واریانس آن را محاسبه کنید.
3\. یک هیستوگرام از داده‌ها رسم کنید.
4\. خطوط میانگین و میانه را روی نمودار رسم کنید.

