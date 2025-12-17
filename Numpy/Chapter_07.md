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

---
مطمئناً! بیایید مفهوم مقادیر ویژه و بردارهای ویژه را با یک مثال ساده‌تر و واضح‌تر توضیح دهیم.

مثال ساده

فرض کنید یک ماتریس A A  داریم:

A=(2112)
A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}


مرحله 1: محاسبه مقادیر ویژه

برای پیدا کردن مقادیر ویژه، باید معادله زیر را حل کنیم:

det(A−λI)=0
\text{det}(A - \lambda I) = 0


که در آن I I  ماتریس واحد است و λ \lambda  مقدار ویژه است. ابتدا ماتریس A−λI A - \lambda I  را محاسبه می‌کنیم:

A−λI=(2−λ112−λ)
A - \lambda I = \begin{pmatrix} 2 - \lambda & 1 \\ 1 & 2 - \lambda \end{pmatrix}


حالا دترمینان این ماتریس را محاسبه می‌کنیم:

det(A−λI)=(2−λ)(2−λ)−(1)(1)=(2−λ)2−1
\text{det}(A - \lambda I) = (2 - \lambda)(2 - \lambda) - (1)(1) = (2 - \lambda)^2 - 1


حالا این معادله را برابر با صفر قرار می‌دهیم:

(2−λ)2−1=0
(2 - \lambda)^2 - 1 = 0


با حل این معادله، داریم:

(2−λ)2=1  ⟹  2−λ=±1
(2 - \lambda)^2 = 1 \implies 2 - \lambda = \pm 1


بنابراین دو مقدار ویژه به دست می‌آید:


2−λ=1 2 - \lambda = 1   ⇒λ=1\Rightarrow \lambda = 1

2−λ=−1 2 - \lambda = -1  ⇒λ=3\Rightarrow \lambda = 3


پس مقادیر ویژه ماتریس A A  عبارتند از λ1=1 \lambda_1 = 1  و λ2=3 \lambda_2 = 3 .

مرحله 2: محاسبه بردارهای ویژه

حالا که مقادیر ویژه را داریم، بیایید بردارهای ویژه متناظر با هر مقدار ویژه را پیدا کنیم.

برای λ1=1 \lambda_1 = 1 :

ما معادله زیر را حل می‌کنیم:

(A−I)v=0
(A - I) \mathbf{v} = 0


که در آن I I  همان ماتریس واحد است. بنابراین:

A−I=(2−1112−1)=(1111)
A - I = \begin{pmatrix} 2-1 & 1 \\ 1 & 2-1 \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}


حالا معادله زیر را حل می‌کنیم:

(1111)(v1v2)=(00)
\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}


این معادله به ما می‌گوید که v1+v2=0 v_1 + v_2 = 0  است، بنابراین می‌توانیم بگوییم:

v2=−v1
v_2 = -v_1


به این ترتیب یکی از بردارهای ویژه برای مقدار ویژه λ1=1 \lambda_1 = 1  می‌تواند به صورت زیر باشد:

v1=(1−1)
\mathbf{v_1} = \begin{pmatrix} 1 \\ -1 \end{pmatrix}


برای λ2=3 \lambda_2 = 3 :

حالا برای مقدار ویژه دوم:

(A−3I)v=0
(A - 3I) \mathbf{v} = 0


بنابراین:

A−3I=(2−3112−3)=(−111−1)
A - 3I = \begin{pmatrix} 2-3 & 1 \\ 1 & 2-3 \end{pmatrix} = \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}


معادله زیر را حل می‌کنیم:

(−111−1)(v1v2)=(00)
\begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}


این معادله به ما می‌گوید که −v1+v2=0 -v_1 + v_2 = 0  است، بنابراین می‌توانیم بگوییم:

v2=v1
v_2 = v_1


به این ترتیب یکی از بردارهای ویژه برای مقدار ویژه λ2=3 \lambda_2 = 3  می‌تواند به صورت زیر باشد:

v2=(11)
\mathbf{v_2} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}


نتیجه‌گیری

بنابراین، برای ماتریس A A :


مقادیر ویژه:

λ1=1 \lambda_1 = 1  با بردار ویژه v1=(1−1) \mathbf{v_1} = \begin{pmatrix} 1 \\ -1 \end{pmatrix} 

λ2=3 \lambda_2 = 3  با بردار ویژه v2=(11) \mathbf{v_2} = \begin{pmatrix} 1 \\ 1 \end{pmatrix} 





مفهوم کلی

به طور کلی، مقادیر ویژه به ما می‌گویند که چگونه یک ماتریس می‌تواند بر روی بردارها تأثیر بگذارد. اگر یک بردار را به یک ماتریس اعمال کنیم و فقط اندازه آن تغییر کند (نه جهت)، آن بردار یک بردار ویژه است و عددی که اندازه آن را تغییر می‌دهد، مقدار ویژه است.

اگر سوال بیشتری دارید یا نیاز به توضیحات بیشتری دارید، خوشحال می‌شوم کمک کنم!




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
