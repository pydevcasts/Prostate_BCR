# 📖 فصل ۶: توابع مهم NumPy برای داده‌کاوی و تحلیل داده‌ها

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. مرتب‌سازی داده‌ها (Sorting)

```python
import numpy as np

arr = np.array([40, 10, 20, 90, 70])

print("Sorted:", np.sort(arr))      # ascending
print("Argsort:", np.argsort(arr))  # indices of sorted elements
```

📌

* `np.sort` داده‌ها را مرتب می‌کند.
* `np.argsort` ایندکس مرتب‌سازی را برمی‌گرداند.

---

### 🔹 ۲. پیدا کردن مقادیر یکتا (Unique)

```python
arr = np.array([1, 2, 2, 3, 4, 4, 5])

unique_values = np.unique(arr)
print("Unique values:", unique_values)
```

📌 `np.unique` تکراری‌ها را حذف کرده و مقادیر یکتا را برمی‌گرداند.

---

### 🔹 ۳. پیدا کردن بیشینه و کمینه

```python
arr = np.array([15, 25, 5, 60, 40])

print("Max:", np.max(arr))
print("Min:", np.min(arr))
print("Argmax:", np.argmax(arr))  # index of max
print("Argmin:", np.argmin(arr))  # index of min
```

📌 `argmax` و `argmin` مکان (ایندکس) بیشینه و کمینه را برمی‌گردانند.

---

### 🔹 ۴. استفاده از `where` برای شرط‌ها

```python
arr = np.array([10, 20, 30, 40, 50])

result = np.where(arr > 25, "High", "Low")
print("Labels:", result)
```

📌 این روش بسیار شبیه `if-else` عمل می‌کند.

---

### 🔹 ۵. تولید اعداد تصادفی

```python
# Random integer array
rand_int = np.random.randint(1, 100, size=10)

# Random float numbers between 0 and 1
rand_float = np.random.rand(5)

# Normal distribution
rand_normal = np.random.randn(5)

print("Random Integers:", rand_int)
print("Random Floats:", rand_float)
print("Random Normal:", rand_normal)
```

📌 `randint`, `rand`, `randn` برای تولید داده‌های تصادفی مختلف کاربرد دارند.

---

### 🔹 ۶. تمرین تصویری 🎨

```python
import matplotlib.pyplot as plt

data = np.random.randn(1000)  # normal distribution

plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title("Histogram of Random Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

📌 این کد یک هیستوگرام از توزیع نرمال رسم می‌کند.

---

### 🔹 ۷. تمرین پیشنهادی 🎯

۱. یک آرایه تصادفی ۲۰ عنصری بسازید.
۲. مقادیر یکتا را پیدا کنید.
۳. بزرگ‌ترین و کوچک‌ترین مقدار و مکان آن‌ها را نمایش دهید.
۴. با استفاده از `where` مقادیر بزرگ‌تر از میانگین را "High" و بقیه را "Low" برچسب بزنید.
۵. یک هیستوگرام از داده‌ها رسم کنید.
