# 📖 فصل ۵: اندیس‌گذاری پیشرفته (Boolean Indexing, Fancy Indexing, Masking)

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. اندیس‌گذاری بولی (Boolean Indexing)

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])

mask = arr > 25   # condition
print("Mask:", mask)
print("Filtered values:", arr[mask])
```

📌 شرط `arr > 25` یک آرایه بولی می‌سازد.
📌 سپس تنها مقادیر درست (True) انتخاب می‌شوند.

---

### 🔹 ۲. ایندکس‌گذاری با شرایط ترکیبی

```python
arr = np.array([5, 15, 25, 35, 45, 55])

filtered = arr[(arr > 20) & (arr < 50)]
print("Values between 20 and 50:", filtered)
```

📌 از عملگرهای `&` (AND) و `|` (OR) برای ترکیب شرایط استفاده می‌کنیم.

---

### 🔹 ۳. Fancy Indexing (ایندکس‌گذاری با لیست ایندکس‌ها)

```python
arr = np.array([100, 200, 300, 400, 500])

indices = [0, 2, 4]   # select first, third, fifth element
print("Selected values:", arr[indices])
```

📌 به جای شرط، می‌توان لیستی از ایندکس‌ها به NumPy داد.

---

### 🔹 ۴. ماسک‌گذاری روی داده‌های دوبعدی

```python
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

mask = arr2d % 2 == 0   # select even numbers
print("Even numbers:", arr2d[mask])
```

📌 ماسک‌ها برای فیلتر کردن داده‌های خاص در ماتریس‌ها عالی عمل می‌کنند.

---

### 🔹 ۵. تمرین تصویری با Boolean Mask 🎨

```python
import matplotlib.pyplot as plt

data = np.random.randint(1, 100, size=50)

mask = data > 50
plt.scatter(range(len(data)), data, c=mask, cmap="bwr", s=100)
plt.title("Values > 50 highlighted")
plt.show()
```

📌 مقادیر بزرگ‌تر از ۵۰ با رنگ متفاوت نمایش داده می‌شوند.

---

### 🔹 ۶. تمرین پیشنهادی 🎯

۱. یک آرایه ۱۰×۱۰ با اعداد تصادفی بین ۱ تا ۱۰۰ بسازید.
۲. همه مقادیر زوج را جدا کنید.
۳. مقادیر بزرگ‌تر از ۷۰ را با استفاده از ماسک پیدا کنید.
۴. فقط سطرهای ۲ و ۴ و ۶ را با fancy indexing انتخاب کنید.
۵. داده‌ها را با یک نمودار scatter نمایش دهید.

