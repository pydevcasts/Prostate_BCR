# 📖 فصل ۸: اندیس‌گذاری پیشرفته و ماسک‌ها در NumPy

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. انتخاب با لیست ایندکس‌ها (Fancy Indexing)

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])

indices = [0, 2, 4]
print("Selected elements:", arr[indices])
```

📌 با دادن لیست ایندکس‌ها می‌توان به چند عنصر خاص دسترسی پیدا کرد.

---

### 🔹 ۲. انتخاب شرطی (Boolean Masking)

```python
arr = np.array([5, 15, 25, 35, 45, 55])

mask = arr > 30
print("Mask:", mask)
print("Filtered values:", arr[mask])
```

📌 شرط `arr > 30` مقادیری که بزرگ‌تر از ۳۰ هستند را انتخاب می‌کند.

---

### 🔹 ۳. ترکیب شرایط

```python
arr = np.arange(1, 21)  # [1,...,20]

mask = (arr % 2 == 0) & (arr > 10)
print("Even numbers > 10:", arr[mask])
```

📌 از `&` برای AND و از `|` برای OR استفاده می‌کنیم.

---

### 🔹 ۴. ماسک روی آرایه دوبعدی

```python
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

mask = arr2d % 2 == 0
print("Even numbers:", arr2d[mask])
```

📌 مقادیر زوج ماتریس جدا می‌شوند.

---

### 🔹 ۵. استفاده از تابع `np.where`

```python
arr = np.array([10, 20, 30, 40, 50])

labels = np.where(arr > 25, "High", "Low")
print("Labels:", labels)
```

📌 `np.where` مثل شرط `if-else` عمل می‌کند.

---

### 🔹 ۶. تمرین تصویری با ماسک 🎨

```python
import matplotlib.pyplot as plt

data = np.random.randint(1, 100, size=50)

mask = data > 50
plt.scatter(range(len(data)), data, c=mask, cmap="bwr", s=80)
plt.title("Highlighting values > 50")
plt.show()
```

📌 مقادیر بزرگ‌تر از ۵۰ با رنگ متفاوت نمایش داده می‌شوند.

---

### 🔹 ۷. تمرین پیشنهادی 🎯

۱. یک آرایه ۱۰×۱۰ با مقادیر تصادفی بین ۱ تا ۱۰۰ بسازید.
۲. همه مقادیر فرد را انتخاب کنید.
۳. فقط سطرهای زوج را نمایش دهید.
۴. مقادیر بزرگ‌تر از میانگین را با ماسک مشخص کنید.
۵. یک scatter plot از داده‌ها رسم کنید و مقادیر بزرگ‌تر از ۵۰ را با رنگ متفاوت نشان دهید.
