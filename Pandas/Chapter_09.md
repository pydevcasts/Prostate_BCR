# 📖 فصل ۹: توابع کاربردی روی داده‌ها (Apply, Map, Applymap)

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. چرا توابع کاربردی مهم هستند؟

در تحلیل داده‌ها، خیلی وقت‌ها نیاز داریم روی ستون‌ها یا کل داده‌ها **عملیات خاصی** انجام بدهیم.
به جای استفاده از حلقه‌ها، در Pandas می‌توانیم از توابعی مثل:

* **map()** → فقط روی **Series**
* **apply()** → روی **Series و DataFrame**
* **applymap()** → فقط روی **DataFrame**

این توابع هم کد را کوتاه می‌کنند و هم سرعت بالاتری دارند.

---

### 🔹 ۲. استفاده از map() روی Series

```python
import pandas as pd

data = pd.Series([1, 2, 3, 4, 5])

# Square each element
print(data.map(lambda x: x**2))
```

📌 خروجی:

```
0     1
1     4
2     9
3    16
4    25
dtype: int64
```

---

### 🔹 ۳. استفاده از apply() روی DataFrame

```python
df = pd.DataFrame({
    "A": [1, 2, 3],
    "B": [4, 5, 6]
})

# Sum of each column
print(df.apply(sum))

# Apply function on each row
print(df.apply(lambda x: x["A"] + x["B"], axis=1))
```

📌 خروجی:

```
A     6
B    15
dtype: int64

0    5
1    7
2    9
dtype: int64
```

---

### 🔹 ۴. استفاده از applymap() روی همه مقادیر DataFrame

```python
# Multiply all values by 10
print(df.applymap(lambda x: x * 10))
```

📌 خروجی:

```
    A   B
0  10  40
1  20  50
2  30  60
```

---

### 🔹 ۵. استفاده ترکیبی

```python
# Normalize data (between 0 and 1)
df_normalized = df.apply(lambda col: (col - col.min()) / (col.max() - col.min()))
print(df_normalized)
```

📌 خروجی:

```
     A    B
0  0.0  0.0
1  0.5  0.5
2  1.0  1.0
```

---

### 🔹 ۶. تمرین عملی

📊 فرض کنید دیتافریم زیر را داریم:

```python
students = pd.DataFrame({
    "Name": ["Ali", "Sara", "Reza"],
    "Score": [85, 90, 78]
})

# 1. Add 5 to all scores
print(students["Score"].map(lambda x: x + 5))

# 2. Convert names to uppercase
print(students["Name"].apply(lambda x: x.upper()))

# 3. Applymap to calculate string length of all values
print(students.applymap(str).applymap(len))
```
