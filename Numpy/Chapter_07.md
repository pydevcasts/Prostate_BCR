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

ماتریس معکوس تنها در صورتی وجود دارد که دترمینان \( A \) نابرابر صفر باشد (\( \text{det}(A) \neq 0 \))..




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

📌 مقدار دترمینان برای بررسی وارون‌پذیری ماتریس مهم است. اگر `det ≠ 0`، ماتریس معکوس‌پذیر است.

---مطمئناً! در زیر خلاصه‌ای از مفهوم مقادیر ویژه و بردارهای ویژه با استفاده از فرمت ریاضی ارائه می‌شود.

### مقادیر ویژه و بردارهای ویژه

برای یک ماتریس مربعی \( A \)، اگر یک عدد \( \lambda \) و یک بردار غیرصفر \( \mathbf{v} \) وجود داشته باشد به طوری که:

$$
A \mathbf{v} = \lambda \mathbf{v}
$$

در اینجا:

- \( \lambda \) **مقدار ویژه** (Eigenvalue) است.
- \( \mathbf{v} \) **بردار ویژه** (Eigenvector) است.

### مثال

فرض کنید ماتریس زیر را داریم:

$$
A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}
$$

#### مرحله 1: محاسبه مقادیر ویژه

برای پیدا کردن مقادیر ویژه، باید دترمینان زیر را محاسبه کنیم:

$$
\text{det}(A - \lambda I) = 0
$$

که در آن \( I \) ماتریس واحد است. این معادله به شکل زیر است:

$$
\begin{vmatrix} 2 - \lambda & 1 \\ 1 & 2 - \lambda \end{vmatrix} = (2 - \lambda)(2 - \lambda) - 1 = 0
$$

با حل این معادله، مقادیر ویژه به دست می‌آید:

$$
\lambda_1 = 1, \quad \lambda_2 = 3
$$

#### مرحله 2: محاسبه بردارهای ویژه

برای هر مقدار ویژه، معادله زیر را حل می‌کنیم:

**برای \( \lambda_1 = 1 \)**:

$$
(A - I) \mathbf{v} = 0 \quad \Rightarrow \quad \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = 0
$$

نتیجه می‌دهد:

$$
\mathbf{v_1} = \begin{pmatrix} 1 \\ -1 \end{pmatrix}
$$

**برای \( \lambda_2 = 3 \)**:

$$
(A - 3I) \mathbf{v} = 0 \quad \Rightarrow \quad \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = 0
$$

نتیجه می‌دهد:

$$
\mathbf{v_2} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}
$$

### نتیجه‌گیری

برای ماتریس \( A \):

- مقادیر ویژه:
  - \( \lambda_1 = 1 \) با بردار ویژه \( \mathbf{v_1} = \begin{pmatrix} 1 \\ -1 \end{pmatrix} \)
  - \( \lambda_2 = 3 \) با بردار ویژه \( \mathbf{v_2} = \begin{pmatrix} 1 \\ 1 \end{pmatrix} \)

این مفهوم به ما می‌گوید که چگونه یک ماتریس می‌تواند بر روی بردارها تأثیر بگذارد. اگر سوال دیگری دارید، خوشحال می‌شوم کمک کنم!



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
import matplotlib.pyplot as plt

A = np.array([[2, 1], [1, 3]])

eig_vals, eig_vecs = np.linalg.eig(A)

# نمایش بردارهای ویژه روی نمودار
origin = [0, 0]
plt.quiver(*origin, eig_vecs[0, :], eig_vecs[1, :], angles='xy', scale_units='xy', scale=1, color=['r','b'])
plt.xlim(-1, 2)
plt.ylim(-1, 2)
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
