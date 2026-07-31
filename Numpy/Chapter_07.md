# 📖 فصل ۷: عملیات ماتریسی در NumPy

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. ضرب ماتریس‌ها

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 3]])

# روش اول: np.dot
dot_product = np.dot(A, B)

# روش دوم: @ operator
matmul_product = A @ B

print("Dot Product:\n", dot_product)
print("Matrix Multiplication:\n", matmul_product)
```

📌 در NumPy می‌توان ضرب ماتریسی را با `np.dot` یا عملگر `@` انجام داد.

---

### 🔹 ۲. ترانهاده (Transpose)

```python
A = np.array([[1, 2, 3], [4, 5, 6]])

print("Original Matrix:\n", A)
print("Transpose:\n", A.T)
```  
        
    

برای یک ماتریس \( 2 \times 2 \):

$$
A = \begin{pmatrix} 
a & b \\ 
c & d 
\end{pmatrix}
$$

## دترمینان

دترمینان آن به صورت زیر محاسبه می‌شود:

$$
\text{det}(A) = ad - bc
$$

## ماتریس هم‌انباشت

ماتریس هم‌انباشت به صورت زیر است:

$$
\text{adj}(A) = \begin{pmatrix} 
d & -b \\ 
-c & a 
\end{pmatrix}
$$

## ماتریس معکوس

بنابراین، ماتریس معکوس به شکل زیر خواهد بود:

$$
A^{-1} = \frac{1}{ad - bc} 
\begin{pmatrix} 
d & -b \\ 
-c & a 
\end{pmatrix}
$$

## نتیجه‌گیری

ماتریس معکوس تنها در صورتی وجود دارد که دترمینان \( A \) نابرابر صفر باشد ..




### 🔹 ۳. معکوس ماتریس (Inverse)

```python
A = np.array([[1, 2], [3, 4]])

inverse_A = np.linalg.inv(A)
print("Inverse of A:\n", inverse_A)
```

📌 تابع `np.linalg.inv` معکوس ماتریس را محاسبه می‌کند (البته فقط برای ماتریس‌های وارون‌پذیر).

---

### 🔹 ۴. دترمینان (Determinant)

```python
A = np.array([[1, 2], [3, 4]])

det_A = np.linalg.det(A)
print("Determinant:", det_A)
```

📌 مقدار دترمینان برای بررسی وارون‌پذیری ماتریس مهم است. اگر `det ≠ 0`، ماتریس معکوس‌پذیر است
---


### 🔹 ۵. مقادیر ویژه و بردارهای ویژه (Eigenvalues & Eigenvectors)

```python
A = np.array([[4, -2], [1, 1]])

eig_vals, eig_vecs = np.linalg.eig(A)

print("Eigenvalues:", eig_vals)
print("Eigenvectors:\n", eig_vecs)
```

📌 `eig` مقدار ویژه و بردارهای ویژه را محاسبه می‌کند. این موضوع در تحلیل داده‌ها (مثلاً PCA) بسیار مهم است.

---

### 🔹 ۶. تمرین تصویری 🎨

```python
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[2, 4], [7, 8]])
eig_vals, eig_vecs = np.linalg.eig(A)

X = [0, 0]
Y = [0, 0]

U = eig_vecs[0, :]
V = eig_vecs[1, :]

plt.figure(figsize=(6, 6))
plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color=['r', 'b'])

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.title("Eigenvectors of Matrix A")
plt.show()

```

📌 در این مثال بردارهای ویژه ماتریس `A` رسم شده‌اند.

---

### 🔹 ۷. تمرین پیشنهادی 🎯

۱. دو ماتریس تصادفی ۳×۳ بسازید و ضرب آن‌ها را محاسبه کنید.
۲. ترانهاده یکی از ماتریس‌ها را پیدا کنید.
۳. دترمینان و معکوس یکی از ماتریس‌ها را محاسبه کنید.
4\. مقادیر ویژه و بردارهای ویژه‌ی آن را نمایش دهید.
