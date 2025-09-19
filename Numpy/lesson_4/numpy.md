# آموزش جامع کتابخانه NumPy 🚀

کتابخانه NumPy (مخفف Numerical Python) یکی از مهم‌ترین و پرکاربردترین کتابخانه‌های پایتون برای انجام محاسبات عددی و علمی است. این کتابخانه به ویژه در زمینه‌های علم داده، یادگیری ماشین و تحلیل داده‌ها بسیار مورد استفاده قرار می‌گیرد. در این مقاله، به بررسی کلیه توابع و ویژگی‌های این کتابخانه می‌پردازیم.

## 1. نصب آناکوندا 🛠️

آناکوندا یک پکیج مدیریتی است که شامل مجموعه‌ای از کتابخانه‌های پایتون برای علم داده و محاسبات عددی می‌باشد. برای نصب آناکوندا، مراحل زیر را دنبال کنید:

- به وب‌سایت آناکوندا مراجعه کرده و نسخه مناسب سیستم عامل خود را دانلود کنید.
- پس از دانلود، فایل را اجرا کرده و مراحل نصب را دنبال کنید.
- در هنگام نصب، به گزینه‌های مشخص شده توجه کنید و تیک آن‌ها را نزنید.

پس از نصب، می‌توانید Anaconda Navigator را از منوی Start پیدا کرده و از آن برای اجرای Jupyter Notebook استفاده کنید.

## مقایسه نامپای با آرایه ⚡

این کد زمان اجرای ضرب عنصر به عنصر دو لیست و دو آرایه NumPy را اندازه‌گیری کرده و نشان می‌دهد که NumPy به طور قابل‌توجهی سریع‌تر است.

```python
import time
import numpy as np  # Import NumPy

# size of arrays and lists
size = 1000000

# declaring lists
list1 = range(size)
list2 = range(size)

# declaring arrays
array1 = np.arange(size)
array2 = np.arange(size)

# list multiplication
initialTime = time.time()
resultantList = [(a * b) for a, b in zip(list1, list2)]

# calculating execution time
print("Time taken by Lists :",
      (time.time() - initialTime),
      "seconds")

# NumPy array multiplication
initialTime = time.time()
resultantArray = array1 * array2

# calculating execution time
print("Time taken by NumPy Arrays :",
      (time.time() - initialTime),
      "seconds")
```

## 2. آرایه‌ها در NumPy 📊

آرایه‌ها در NumPy ساختار داده اصلی این کتابخانه هستند و مشابه لیست‌ها در پایتون عمل می‌کنند، اما قابلیت‌های بیشتری دارند. برای ایجاد یک آرایه، از تابع `np.array` استفاده می‌شود:

```python
import numpy as np
myPythonList = [1, 9, 8, 3]
numpy_array_from_list = np.array(myPythonList)
print(numpy_array_from_list)  # خروجی: [1 9 8 3]
```

## 3. Broadcasting یا انتشار همگانی 🌍

Broadcasting به نحوه‌ی اعمال عملگرها بر روی آرایه‌هایی با ابعاد متفاوت اشاره دارد. به طور کلی، آرایه کوچکتر به اندازه‌ی آرایه بزرگتر گسترش می‌یابد تا بتوانند با هم محاسبات انجام دهند.

### مثال:
```python
import numpy as np
a = np.array([1, 2, 3])
b = 2
print(a + b)  # خروجی: [3 4 5]
```

## 4. شکل آرایه (Shape of Array) 📐

برای به دست آوردن شکل یک آرایه، می‌توانید از ویژگی `shape` استفاده کنید:

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape)  # خروجی: (2, 3)
```

## 5. آرایه با مقادیر صفر و یک 🔢

برای ایجاد آرایه‌هایی با مقادیر صفر یا یک، می‌توانید از توابع `np.zeros` و `np.ones` استفاده کنید:

```python
zeros_array = np.zeros((2, 2))
ones_array = np.ones((2, 2))
```

## 6. تغییر ابعاد آرایه 🔄

برای تغییر ابعاد آرایه از `reshape` و برای مسطح کردن آرایه از `flatten` استفاده می‌شود:

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
reshaped_array = a.reshape(3, 2)
flattened_array = a.flatten()
```

## 7. اتصال آرایه‌ها (Stacking) 🔗

برای اتصال چندین آرایه می‌توانید از توابع `np.vstack`، `np.hstack` و `np.concatenate` استفاده کنید:

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
vertical_stack = np.vstack((a, b))
horizontal_stack = np.hstack((a, b))
```

## 8. تقسیم آرایه‌ها (Splitting) ✂️

برای تقسیم آرایه‌ها می‌توانید از توابع `np.hsplit` و `np.vsplit` استفاده کنید:

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
horizontal_split = np.hsplit(a, 2)
vertical_split = np.vsplit(a, 2)
```

## 9. پیمایش آرایه‌ها 🔍

برای پیمایش آرایه‌ها، می‌توانید از حلقه‌های تودرتو یا تابع `nditer` استفاده کنید:

```python
for x in np.nditer(a):
    print(x)
```

## 10. جبر خطی در NumPy 📏

ماژول جبر خطی در NumPy شامل توابعی برای انجام محاسبات مختلف، از جمله دترمینان و معکوس ماتریس است. برای درک بهتر دترمینان، یک مثال ساده با یک ماتریس ۲x۲ را بررسی می‌کنیم:

فرض کنید ماتریس A به صورت زیر باشد:

A =[[a , b],[c , d]]

دترمینان آن به شکل زیر محاسبه می‌شود:

(A) = ad - bc

```python
A = np.array([[3, 2], [1, 4]])
determinant = np.linalg.det(A)
print(determinant)  # 10.0
```

## 11.ایندکس بزرگ‌ترین مقدار در آرایه 🎲

```python
arr = np.array([1, 3, 2, 5, 4])
index_of_max = np.argmax(arr)

print(index_of_max)  # خروجی: 3
```

## 12. کار با تاریخ و زمان 📅

NumPy نوع داده‌ای به نام `datetime64` برای کار با تاریخ و زمان دارد:

```python
today = np.datetime64('2025-05-21')
# np.datetime64('2025-05-21')
np.datetime_as_string(np.datetime64("2025-10-10"))
# np.str_('2025-10-10')
```

## 13. تبدیل داده‌ها به آرایه 🔄

برای تبدیل داده‌های ورودی به آرایه از تابع `asarray` استفاده می‌شود:

```python
data = [1, 2, 3]
array = np.asarray(data)
```

در کتابخانه NumPy در پایتون، `np.array` و `np.asarray` هر دو برای ایجاد آرایه‌ها (arrays) استفاده می‌شوند، اما تفاوت‌هایی در نحوه‌ی کارکرد آنها وجود دارد:

```python
# Using np.array
a = np.array([1, 2, 3])
b = np.array(a)  # A new copy is created here

# Using np.asarray
c = np.asarray([1, 2, 3])
d = np.asarray(a)  # No new copy is created here; it references the original array 'a'

print(a is b)  # False, because 'b' is a new copy of 'a'
print(a is d)  # True, because 'd' references the same array as 'a'
```
اگر شما به یک کپی از داده‌ها نیاز دارید، از np.array استفاده کنید. اگر فقط می‌خواهید داده‌ها را به یک آرایه NumPy تبدیل کنید و از کپی کردن آنها اجتناب کنید، np.asarray انتخاب بهتری است.

## 14. ایجاد آرایه با `numpy.arange` 📏

این تابع آرایه‌ای از اعداد در یک بازه مشخص ایجاد می‌کند:

```python
array = np.arange(1, 10, 2)  # خروجی: [1, 3, 5, 7, 9]
```

## 15. تولید دنباله با `numpy.linspace` و `numpy.logspace` 📈

این توابع دنباله‌ای از اعداد با فاصله‌ی مساوی تولید می‌کنند:

```python
linspace_array = np.linspace(1, 10, num=5)  # خروجی: [1.  3.25 5.5  7.75 10.]
logspace_array = np.logspace(1, 3, num=3)  # خروجی: [  10.  100. 1000.]
```

## 16. اندیس‌دهی و برش آرایه‌ها 🔍

برای برش آرایه‌ها از اندیس‌ها استفاده می‌شود:

```python
array = np.array([[1, 2, 3], [4, 5, 6]])
print(array[0, :])  # خروجی: [1 2 3]
```

## 17. توابع آماری NumPy 📊

برای محاسبه میانگین و انحراف معیار (Standard Deviation) یک آرایه در NumPy، می‌توانید از توابع `np.mean` و `np.std` استفاده کنید. در زیر یک مثال ساده برای نشان دادن چگونگی استفاده از این توابع آورده شده است.

### مثال:

فرض کنید می‌خواهید میانگین و انحراف معیار یک آرایه عددی را محاسبه کنید:

```python
import numpy as np

# Define an array
array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Calculate the mean
mean = np.mean(array)

# Calculate the standard deviation
std_dev = np.std(array)

print("Mean of the array:", mean)  # Output: 5.5
print("Standard deviation of the array:", std_dev)  # Output: 2.8722813232690143
```

### توضیحات:

**میانگین**: میانگین آرایه به سادگی مجموع همه مقادیر تقسیم بر تعداد آن‌ها است. در این مثال، میانگین آرایه از ۱ تا ۱۰ برابر با ۵.۵ است.
  
**انحراف معیار**: انحراف معیار معیاری برای اندازه‌گیری پراکندگی مقادیر از میانگین است. در این مثال، انحراف معیار تقریباً برابر با ۲.۸۷ است که نشان می‌دهد مقادیر چقدر از میانگین فاصله دارند.

## 18. ضرب ماتریسی ➕ 

برای انجام ضرب ماتریسی از `numpy.dot` استفاده می‌شود:

```python
import numpy as np

# Define two vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Calculate the dot product
dot_product = np.dot(a, b)

print("Dot product of a and b:", dot_product)  # Output: 32

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Perform matrix multiplication
result = np.dot(a, b)

print(result)
```
C[0,0] = (1 * 5) + (2 * 7) = 5 + 14 = 19

C[0,1] = (1 * 6) + (2 * 8) = 6 + 16 = 22

C[1,0] = (3 * 5) + (4 * 7) = 15 + 28 = 43

C[1,1] = (3 * 6) + (4 * 8) = 18 + 32 = 50


### Result:
The resulting matrix `C` will be:
```
[[19, 22],
 [43, 50]]
```

## 19. تفاوت بین Copy و View 🔄

تفاوت میان کپی و ویو در این است که کپی یک آرایه جدید است، در حالی که ویو تنها یک نمای از آرایه اصلی است:

```python
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[0] = 42
print(arr)
print(x)

#[42  2  3  4  5]
# [1 2 3 4 5]
```

```python
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
x[0] = 31
print(arr)
print(x)
# [31  2  3  4  5]
# [31  2  3  4  5]
```

## 20. تولید اعداد تصادفی 🎲

### a. اعداد تصادفی یکنواخت

برای تولید اعداد تصادفی در بازه \([0, 1)\):

```python
# Generate 5 uniform random numbers
random_numbers_uniform = np.random.rand(5)
print("Random numbers (uniform):", random_numbers_uniform)
```

## 21. تولید اعداد تصادفی از توزیع نرمال (Gaussian) 🎲

برای تولید اعداد تصادفی از توزیع نرمال با میانگین و انحراف معیار مشخص:

```python
# Generate 5 random numbers from a normal distribution with mean 0 and standard deviation 1
random_numbers_normal = np.random.normal(0, 1, 5)
print("Random numbers (normal distribution):", random_numbers_normal)
```

## 22. نمونه‌گیری از یک آرایه 🎲

برای نمونه‌گیری تصادفی از یک آرایه:

```python
# Define an array
array = np.array([10, 20, 30, 40, 50])

# Randomly sample from the array
random_sample = np.random.choice(array, size=3, replace=False)  # without replacement
print("Random sample from array:", random_sample)
```

## 23. تولید اعداد تصادفی صحیح 🎲

برای تولید اعداد صحیح تصادفی در یک بازه مشخص:

```python
# Generate 5 random integers between 1 and 100
random_integers = np.random.randint(1, 101, size=5)
print("Random integers:", random_integers)

random_integers_limit = np.random.randint(1, [3, 5, 10])
print("Random integers:", random_integers_limit)

```
### 💡 توضیح:

این دستور از تابع `np.random.randint` استفاده می‌کند، اما برخلاف حالت معمول که یک عدد به عنوان **حد بالایی** مشخص می‌شود، اینجا از **آرایه‌ای** به عنوان حد بالا استفاده شده است.

فرم کلی دستور:

```python
np.random.randint(low, high, size)
```

در اینجا:

* `low = 1`
* `high = [3, 5, 10]`
* `size` به صورت ضمنی از طول آرایه `high` تعیین می‌شود: یعنی `size=3`

بنابراین:

🔹 عدد اول از بازه `[1, 3)`
🔹 عدد دوم از بازه `[1, 5)`
🔹 عدد سوم از بازه `[1, 10)`

در نتیجه، یک آرایه‌ی شامل **۳ عدد تصادفی** تولید می‌شود که هرکدام از بازه‌ی متفاوتی آمده‌اند.

📌 خروجی ممکن است هر بار متفاوت باشد.


## 24. تنظیم بذر تصادفی 🎲

برای تولید اعداد تصادفی قابل تکرار، می‌توانید بذر تصادفی را تنظیم کنید:

```python
# Set a random seed
np.random.seed(42)

# Generate 5 random numbers using the seed
random_numbers_seeded = np.random.rand(5)
print("Random numbers with seed:", random_numbers_seeded)
```

## 25. تولید ماتریس تصادفی 🎲

برای تولید یک ماتریس تصادفی:

```python
# Generate a 3x3 matrix with uniform random numbers
random_matrix = np.random.rand(3, 3)
print("Random matrix:\n", random_matrix)
```

## 26. تولید اعداد تصادفی با توزیع یکنواخت در یک بازه مشخص 🎲

برای تولید اعداد تصادفی یکنواخت در یک بازه خاص:

```python
# Generate 5 uniform random numbers between 5 and 10
random_uniform_range = np.random.uniform(5, 10, size=5)
print("Random numbers (uniform in range 5 to 10):", random_uniform_range)
```

## 27. استفاده از متدهای `.max()` و `.min()` 📏

```python
# Create an array
array = np.array([10, 20, 5, 30, 15])

# Finding maximum value
max_value = np.max(array)
print("Maximum value:", max_value)  # Output: 30

# Finding minimum value
min_value = np.min(array)
print("Minimum value:", min_value)  # Output: 5
```

## 28. پیدا کردن مقادیر حداکثر و حداقل در یک آرایه دوبعدی 📊

اگر یک آرایه 2 بعدی دارید، می‌توانید محور (axis) را مشخص کنید تا مقادیر حداکثر یا حداقل را پیدا کنید.

```python
# Create a 2D array
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Finding the maximum value in the entire matrix
max_value_matrix = np.max(matrix)
print("Maximum value in matrix:", max_value_matrix)  # Output: 9

# Finding the minimum value in the entire matrix
min_value_matrix = np.min(matrix)
print("Minimum value in matrix:", min_value_matrix)  # Output: 1

# Finding the maximum value along axis 0 (columns)
max_value_axis0 = np.max(matrix, axis=0)
print("Maximum values along axis 0 (columns):", max_value_axis0)  # Output: [7 8 9]

# Finding the minimum value along axis 1 (rows)
min_value_axis1 = np.min(matrix, axis=1)
print("Minimum values along axis 1 (rows):", min_value_axis1)  # Output: [1 4 7]
```

