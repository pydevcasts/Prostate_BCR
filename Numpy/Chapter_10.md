# 📖 فصل ۱۰: عملیات ورودی و خروجی در NumPy

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. ذخیره و بارگذاری با فرمت NPY

```python
import numpy as np

arr = np.arange(1, 11)

# Save array to .npy file
np.save("my_array.npy", arr)

# Load array from .npy file
loaded = np.load("my_array.npy")

print("Loaded array:", loaded)
```

📌 فرمت **NPY** مخصوص NumPy است و سریع‌ترین روش ذخیره و بارگذاری داده‌ها محسوب می‌شود.

---

### 🔹 ۲. ذخیره و بارگذاری چند آرایه با NPZ

```python
arr1 = np.arange(5)
arr2 = np.arange(10, 15)

# Save multiple arrays
np.savez("multi_arrays.npz", first=arr1, second=arr2)

# Load arrays
data = np.load("multi_arrays.npz")
print("First:", data["first"])
print("Second:", data["second"])
```

📌 فایل **NPZ** در واقع یک zip است که شامل چند آرایه می‌شود.

---

### 🔹 ۳. ذخیره و بارگذاری در فایل متنی

```python
arr = np.array([[1.5, 2.3, 3.1],
                [4.2, 5.5, 6.8]])

# Save to text file
np.savetxt("data.txt", arr, fmt="%.2f", delimiter=",")

# Load from text file
loaded_txt = np.loadtxt("data.txt", delimiter=",")
print("Loaded from txt:\n", loaded_txt)
```

📌 با `fmt` قالب نمایش اعداد تعیین می‌شود.

---

### 🔹 ۴. ذخیره و بارگذاری فایل CSV

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Save as CSV
np.savetxt("data.csv", arr, fmt="%d", delimiter=",")

# Load CSV
loaded_csv = np.loadtxt("data.csv", delimiter=",", dtype=int)
print("Loaded CSV:\n", loaded_csv)
```

📌 برای داده‌های جدولی مثل اکسل یا pandas می‌توان CSV استفاده کرد.

---

### 🔹 ۵. ترکیب NumPy با Matplotlib برای ذخیره داده‌ها 📊

```python
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 50)
y = np.sin(x)

data = np.column_stack((x, y))
np.savetxt("sin_wave.csv", data, delimiter=",", header="x,y", comments="")

plt.plot(x, y)
plt.title("Sin Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.show()
```

📌 هم داده ذخیره شد، هم نمودار رسم شد.

---

### 🔹 ۶. تمرین پیشنهادی 🎯

۱. یک آرایه ۱۰×۱۰ از اعداد تصادفی بسازید و آن را در فایل NPY ذخیره کنید.
۲. همان آرایه را در یک فایل CSV ذخیره کنید.
۳. دوباره آن را از CSV بخوانید و میانگین سطرها را محاسبه کنید.
۴. یک فایل NPZ شامل دو آرایه ذخیره کنید.
۵. داده‌های خروجی را با Matplotlib ترسیم کنید.

