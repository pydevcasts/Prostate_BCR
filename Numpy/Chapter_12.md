
# 📖 فصل ۱۲: جبر خطی در NumPy

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. معرفی ماژول `numpy.linalg`

در NumPy یک زیرماژول به نام `linalg` وجود دارد که مخصوص **محاسبات جبر خطی** است.
این بخش شامل دترمینان، معکوس ماتریس، مقادیر ویژه (Eigenvalues)، حل دستگاه معادلات و … می‌شود.

---

### 🔹 ۲. ضرب ماتریسی

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# Matrix multiplication
C = np.dot(A, B)
print("A * B =\n", C)

# یا معادل
print("Using @ operator:\n", A @ B)
```

📌 برای ضرب ماتریسی باید ابعاد سازگار باشند (تعداد ستون‌های A برابر با تعداد سطرهای B).

---

### 🔹 ۳. دترمینان ماتریس

```python
det_A = np.linalg.det(A)
print("Determinant of A:", det_A)
```

📌 دترمینان در ریاضیات کاربرد زیادی دارد؛ مثلاً اگر صفر باشد یعنی ماتریس معکوس‌پذیر نیست.

---

### 🔹 ۴. معکوس ماتریس

```python
inv_A = np.linalg.inv(A)
print("Inverse of A:\n", inv_A)

# Check: A * inv_A = I
print("A * inv_A:\n", np.dot(A, inv_A))
```

📌 حاصل ضرب ماتریس در معکوس آن برابر با **ماتریس همانی (I)** می‌شود.

---

### 🔹 ۵. حل دستگاه معادلات خطی

فرض کنید دستگاه زیر را داریم:

$$
2x + y = 5
$$

$$
x - y = 1
$$

کد:

```python
# Coefficients matrix
coef = np.array([[2, 1],
                 [1, -1]])

# Constants vector
const = np.array([5, 1])

# Solve system
solution = np.linalg.solve(coef, const)
print("Solution (x, y):", solution)
```

📌 خروجی مقدار $x$ و $y$ خواهد بود.

---

### 🔹 ۶. مقادیر ویژه و بردارهای ویژه

```python
vals, vecs = np.linalg.eig(A)

print("Eigenvalues:", vals)
print("Eigenvectors:\n", vecs)
```

📌 مقادیر و بردارهای ویژه در هوش مصنوعی و الگوریتم‌هایی مثل PCA کاربرد زیادی دارند.

---

### 🔹 ۷. مثال ترکیبی: تحلیل ماتریس با NumPy و Matplotlib

```python
import matplotlib.pyplot as plt

M = np.array([[4, -2],
              [1, 1]])

vals, vecs = np.linalg.eig(M)

print("Eigenvalues:", vals)
print("Eigenvectors:\n", vecs)

plt.quiver([0, 0], [0, 0], vecs[0, :], vecs[1, :],
           angles="xy", scale_units="xy", scale=1, color=["r", "b"])

plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid()
plt.title("Eigenvectors of Matrix M")
plt.show()
```

📌 در این مثال بردارهای ویژه به صورت فلش روی صفحه ترسیم می‌شوند.

---

### 🔹 ۸. تمرین پیشنهادی 🎯

۱. یک ماتریس ۳×۳ بسازید.
۲. دترمینان و معکوس آن را محاسبه کنید.
۳. یک دستگاه معادلات ۳ متغیره با NumPy حل کنید.
۴. مقادیر و بردارهای ویژه ماتریس را رسم کنید.
