# 📖 فصل ۲: ساخت و کار با آرایه‌ها در NumPy

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. ایجاد آرایه با `np.array()`

```python
import numpy as np

# Create 1D array
arr1 = np.array([1, 2, 3, 4, 5])

# Create 2D array (matrix)
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

print("1D Array:", arr1)
print("2D Array:\n", arr2)
```

📌 توضیح:

* `arr1` یک آرایه یک‌بعدی است.
* `arr2` یک ماتریس دوبعدی است.

---

### 🔹 ۲. استفاده از توابع کمکی برای ساخت آرایه‌ها

```python
# Create array with range of numbers
arr = np.arange(0, 10, 2)  # from 0 to 10 step 2

# Create array with equally spaced numbers
arr_lin = np.linspace(0, 1, 5)  # 5 numbers between 0 and 1

# Create array of zeros
zeros = np.zeros((2, 3))

# Create array of ones
ones = np.ones((3, 2))

print("arange:", arr)
print("linspace:", arr_lin)
print("zeros:\n", zeros)
print("ones:\n", ones)
```

📌 `np.arange` مثل `range` در پایتون عمل می‌کند.
📌 `np.linspace` مقادیر مساوی بین یک بازه تولید می‌کند.
📌 `zeros` و `ones` برای ایجاد آرایه پرشده با صفر یا یک استفاده می‌شود.

---

### 🔹 ۳. بررسی ویژگی‌های آرایه‌ها

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print("Shape:", arr.shape)   # dimensions
print("Size:", arr.size)     # total elements
print("NDIM:", arr.ndim)     # number of dimensions
print("Dtype:", arr.dtype)   # data type
```

📌

* `shape` → ابعاد آرایه
* `size` → تعداد کل عناصر
* `ndim` → تعداد بعد
* `dtype` → نوع داده

---

### 🔹 ۴. تغییر نوع داده (Type Casting)

```python
arr = np.array([1.5, 2.3, 3.7])

# Convert float to integer
arr_int = arr.astype(int)

print("Original:", arr)
print("Converted:", arr_int)
```

📌 خیلی وقت‌ها برای کاهش حافظه یا نیاز محاسباتی باید نوع داده تغییر کند.

---

### 🔹 ۵. تمرین پیشنهادی 🎯

۱. یک آرایه ۳×۳ با اعداد تصادفی بین ۱ تا ۱۰ بسازید.
۲. نوع داده‌ی آن را بررسی کنید.
۳. آن را به عدد صحیح (int) تبدیل کنید.
۴. تعداد بعد و شکل آن را چاپ کنید.

