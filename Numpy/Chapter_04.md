
# 📖 فصل ۴: عملیات ریاضی و آماری روی آرایه‌ها

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. عملیات ریاضی پایه

```python
import numpy as np

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([10, 20, 30, 40])

print("Addition:", arr1 + arr2)
print("Subtraction:", arr2 - arr1)
print("Multiplication:", arr1 * arr2)
print("Division:", arr2 / arr1)
print("Power:", arr1 ** 2)
```

📌 عملگرها به صورت **عنصری (element-wise)** عمل می‌کنند.

---

### 🔹 ۲. توابع ریاضی داخلی NumPy

```python
arr = np.array([1, 4, 9, 16, 25])

print("Square root:", np.sqrt(arr))
print("Exponential:", np.exp(arr))
print("Logarithm:", np.log(arr))
print("Sine:", np.sin(arr))
```

📌 توابع ریاضی پرکاربرد به‌صورت مستقیم روی آرایه‌ها کار می‌کنند.

---

### 🔹 ۳. عملیات آماری

```python
arr = np.array([10, 20, 30, 40, 50])

print("Mean:", np.mean(arr))       # average
print("Median:", np.median(arr))   # middle value
print("Standard Deviation:", np.std(arr))  # spread of data
print("Variance:", np.var(arr))    # squared std
print("Min:", np.min(arr))
print("Max:", np.max(arr))
print("Sum:", np.sum(arr))
```

📌 این توابع برای تحلیل داده‌ها بسیار کاربردی هستند.

---

### 🔹 ۴. عملیات روی محورهای خاص (Axis)

```python
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print("Sum over rows:", np.sum(arr2d, axis=0))
print("Sum over columns:", np.sum(arr2d, axis=1))
```

📌 `axis=0` → روی ستون‌ها
📌 `axis=1` → روی ردیف‌ها

---

### 🔹 ۵. تمرین تصویری با Matplotlib 🎨

```python
import matplotlib.pyplot as plt

data = np.random.randint(1, 100, size=(10, 5))

plt.imshow(data, cmap='viridis', aspect='auto')
plt.colorbar(label="Values")
plt.title("Matrix Visualization")
plt.show()
```

📌 با `imshow` می‌توانیم ماتریس‌ها یا داده‌های آماری را به شکل تصویری نمایش دهیم.

---

### 🔹 ۶. تمرین پیشنهادی 🎯

۱. یک آرایه تصادفی ۵×۵ بسازید.
۲. میانگین، واریانس و انحراف معیار را محاسبه کنید.
۳. مجموع هر ستون را به‌دست بیاورید.
۴. یک نمودار تصویری از این ماتریس رسم کنید.
