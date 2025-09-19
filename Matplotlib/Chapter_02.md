## 📖 فصل ۲: نمودارهای پایه‌ای

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. نمودار خطی (Line Plot)

این ساده‌ترین و رایج‌ترین نمودار است.

```python
import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x, y, color='blue', marker='o', linestyle='--')
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Line Plot Example")
plt.show()
```

📌 این کد یک **نمودار خطی با نقاط دایره‌ای** رسم می‌کند.

---

### 🔹 ۲. نمودار میله‌ای (Bar Chart)

```python
categories = ["A", "B", "C", "D", "E"]
values = [5, 7, 3, 8, 4]

plt.bar(categories, values, color='orange')
plt.xlabel("Categories")
plt.ylabel("Values")
plt.title("Bar Chart Example")
plt.show()
```

📌 نمودار میله‌ای برای مقایسه دسته‌ها (Categories) بسیار کاربردی است.

---

### 🔹 ۳. نمودار میله‌ای افقی (Horizontal Bar)

```python
plt.barh(categories, values, color='green')
plt.xlabel("Values")
plt.ylabel("Categories")
plt.title("Horizontal Bar Chart Example")
plt.show()
```

📌 زمانی که تعداد دسته‌ها زیاد باشد یا اسامی طولانی باشند، **barh** خواناتر است.

---

### 🔹 ۴. نمودار پراکندگی (Scatter Plot)

```python
import numpy as np

x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x, y, color='red', marker='x')
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Scatter Plot Example")
plt.show()
```

📌 Scatter Plot برای نمایش **رابطه بین دو متغیر** استفاده می‌شود.

---

### 🔹 ۵. نمودار دایره‌ای (Pie Chart)

```python
sizes = [30, 25, 20, 15, 10]
labels = ["A", "B", "C", "D", "E"]
colors = ["gold", "lightblue", "lightgreen", "pink", "orange"]

plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
plt.title("Pie Chart Example")
plt.show()
```

📌 نمودار دایره‌ای برای نمایش **درصد سهم دسته‌ها** کاربرد دارد.

---

### 🔹 نتیجه‌گیری فصل ۲

* نمودار خطی برای نمایش تغییرات.
* نمودار میله‌ای برای مقایسه دسته‌ها.
* نمودار پراکندگی برای روابط بین متغیرها.
* نمودار دایره‌ای برای درصد سهم‌ها.

