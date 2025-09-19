# 📖 فصل ۳: عملیات پایه‌ای روی آرایه‌ها (Indexing, Slicing, Reshape, Concatenate)

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. ایندکس‌گذاری (Indexing)

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])

print("First element:", arr[0])
print("Last element:", arr[-1])
```

📌 ایندکس در آرایه‌ها از صفر شروع می‌شود.
📌 ایندکس منفی یعنی شمارش از آخر.

---

### 🔹 ۲. برش (Slicing)

```python
arr = np.array([10, 20, 30, 40, 50, 60, 70])

print("Slice [1:4]:", arr[1:4])    # from index 1 to 3
print("Slice [:4]:", arr[:4])      # from beginning to index 3
print("Slice [3:]:", arr[3:])      # from index 3 to end
print("Step slicing [::2]:", arr[::2])  # every 2 elements
```

📌 ساختار برش: `arr[start:end:step]`

---

### 🔹 ۳. ایندکس‌گذاری در آرایه‌های چندبعدی

```python
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print("Element at (0,0):", arr2d[0, 0])
print("Element at (2,1):", arr2d[2, 1])
print("Row 1:", arr2d[1, :])
print("Column 2:", arr2d[:, 2])
```

📌 برای دسترسی به عناصر دوبعدی: `arr[row, col]`

---

### 🔹 ۴. تغییر شکل (Reshape)

```python
arr = np.arange(1, 13)  # [1,2,...,12]

reshaped = arr.reshape(3, 4)

print("Original:", arr)
print("Reshaped (3x4):\n", reshaped)
```

📌 `reshape` فقط زمانی ممکن است که تعداد عناصر تغییر نکند.

---

### 🔹 ۵. الحاق آرایه‌ها (Concatenate & Stack)

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

concat = np.concatenate((arr1, arr2))
vstack = np.vstack((arr1, arr2))
hstack = np.hstack((arr1, arr2))

print("Concatenate:", concat)
print("Vertical Stack:\n", vstack)
print("Horizontal Stack:", hstack)
```

📌

* `concatenate` → اتصال ساده
* `vstack` → چسباندن عمودی (بالا و پایین)
* `hstack` → چسباندن افقی (چپ و راست)

---

### 🔹 ۶. تمرین پیشنهادی 🎯

۱. یک آرایه ۲×۵ بسازید.
۲. عنصر سطر ۱ ستون ۳ را چاپ کنید.
۳. با استفاده از برش، ستون دوم را جدا کنید.
۴. آن را به شکل ۵×۲ تغییر دهید.
۵. با یک آرایه‌ی دیگر الحاق کنید.

