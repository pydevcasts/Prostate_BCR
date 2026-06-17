# 📖 فصل ۹: ترکیب، تقسیم و تغییر شکل آرایه‌ها

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. الحاق آرایه‌ها با `concatenate`

```python
import numpy as np

arr1 = np.array([1, 2, 3])



concat = np.concatenate((arr1, arr2))
print("Concatenate:", concat)
```

📌 `concatenate` آرایه‌ها را به‌صورت خطی به هم وصل می‌کند.

---

### 🔹 ۲. اتصال عمودی و افقی

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

vstack = np.vstack((A, B))
hstack = np.hstack((A, B))

print("Vertical Stack:\n", vstack)
print("Horizontal Stack:\n", hstack)
```

📌

* `vstack` → اتصال عمودی (بالا و پایین)
* `hstack` → اتصال افقی (چپ و راست)

---

### 🔹 ۳. تقسیم آرایه‌ها با `split`

```python
arr = np.arange(1, 10)

split_arr = np.split(arr, 3)
print("Splitted arrays:", split_arr)
```

📌 آرایه به سه بخش مساوی تقسیم می‌شود.

---

### 🔹 ۴. تغییر شکل با `reshape`

```python
arr = np.arange(1, 13)

reshaped = arr.reshape(3, 4)
print("Reshaped (3x4):\n", reshaped)
```

📌 تغییر شکل فقط در صورتی ممکن است که تعداد عناصر تغییر نکند.

---

### 🔹 ۵. تکرار داده‌ها با `tile` و `repeat`

```python
arr = np.array([1, 2, 3])

# Repeat each element
rep = np.repeat(arr, 2)

# Tile the whole array
tiled = np.tile(arr, 3)

print("Repeat:", rep)
print("Tile:", tiled)
```

📌 `repeat` → تکرار عناصر
📌 `tile` → تکرار کل آرایه

---

### 🔹 ۶. تمرین تصویری 🎨

```python
import matplotlib.pyplot as plt

arr = np.arange(1, 7).reshape(2, 3)

plt.imshow(arr, cmap="Blues", aspect="auto")
plt.colorbar(label="Values")
plt.title("Reshaped Array Visualization")
plt.show()
```

📌 در اینجا آرایه بازشکل داده شده با رنگ نمایش داده می‌شود.

---

### 🔹 ۷. تمرین پیشنهادی 🎯

۱. دو آرایه ۲×۲ بسازید و به صورت عمودی و افقی به هم بچسبانید.
۲. یک آرایه ۱۲ عنصری بسازید و به ۳ بخش تقسیم کنید.
۳. آن را به شکل ۲×۶ تغییر دهید.
۴. آرایه‌ای بسازید و عناصرش را ۳ بار تکرار کنید.
۵. کل آرایه را ۴ بار تکرار کنید.

