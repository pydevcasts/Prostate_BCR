# 📖 فصل ۱: مقدمه و نصب NumPy

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. NumPy چیست؟

کتابخانه‌ی **NumPy (Numerical Python)** یکی از مهم‌ترین ابزارهای دنیای علم داده و یادگیری ماشین است.
این کتابخانه برای کار با **آرایه‌های چندبعدی (ndarray)** و انجام محاسبات عددی بهینه طراحی شده است.

✨ چرا NumPy اینقدر مهم است؟

* سرعت بسیار بالاتر نسبت به لیست‌های پایتون 🚀
* پشتیبانی از عملیات ریاضی و آماری پیشرفته ➕✖️
* استفاده به‌عنوان پایه در کتابخانه‌های دیگر (مثل Pandas، Scikit-Learn، TensorFlow) 🏗️
* قابلیت یکپارچگی با زبان‌های C و Fortran 🔧

---

### 🔹 ۲. نصب NumPy

برای نصب NumPy کافیست در محیط **Terminal یا Anaconda Prompt** دستور زیر را اجرا کنید:

```bash
pip install numpy
```

یا اگر از **Anaconda** استفاده می‌کنید:

```bash
conda install numpy
```

---

### 🔹 ۳. اولین کد با NumPy

```python
import numpy as np   # Import NumPy library

# Create a simple array
arr = np.array([1, 2, 3, 4, 5])

# Print array
print("Array:", arr)

# Print type of arr
print("Type:", type(arr))

# Print shape of arr
print("Shape:", arr.shape)
```

📌 توضیح:

* با `np.array()` یک آرایه ساختیم.
* خروجی نوع داده رو `numpy.ndarray` نشون میده.
* `shape` تعداد بعد و اندازه آرایه رو مشخص می‌کنه.

---

### 🔹 ۴. مقایسه لیست پایتون با NumPy

```python
import numpy as np
import time

# Create a Python list
py_list = list(range(1_000_000))

# Create a NumPy array
np_array = np.arange(1_000_000)

# Sum using Python
start = time.time()
sum_py = sum(py_list)
end = time.time()
print("Python list sum:", sum_py, "Time:", end - start)

# Sum using NumPy
start = time.time()
sum_np = np.sum(np_array)
end = time.time()
print("NumPy array sum:", sum_np, "Time:", end - start)
```

📊 نتیجه:

* جمع لیست پایتون خیلی کندتره.
* جمع NumPy با سرعت بسیار بالا انجام میشه چون از **C زیرساختی** استفاده می‌کنه.

