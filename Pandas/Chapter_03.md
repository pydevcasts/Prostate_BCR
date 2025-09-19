# 📖 فصل ۳: آشنایی با DataFrame در Pandas

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. معرفی DataFrame

**DataFrame** مهم‌ترین ساختار داده‌ای در Pandas است.
می‌توانید آن را مثل یک **جدول داده‌ای (مشابه Excel)** در نظر بگیرید که شامل:

* ردیف‌ها (Index)
* ستون‌ها (Columns)
* داده‌ها (Values)

است.

---

### 🔹 ۲. ساخت DataFrame از دیکشنری

```python
import pandas as pd

data = {
    "Name": ["Ali", "Sara", "Reza"],
    "Age": [25, 22, 30],
    "Score": [85, 90, 78]
}

df = pd.DataFrame(data)
print(df)
```

📌 خروجی:

```
   Name  Age  Score
0   Ali   25     85
1  Sara   22     90
2  Reza   30     78
```

---

### 🔹 ۳. دسترسی به ستون‌ها

```python
print(df["Name"])   # دسترسی به ستون Name
print(df.Age)       # روش دیگر دسترسی
```

📌 خروجی:

```
0     Ali
1    Sara
2    Reza
Name: Name, dtype: object

0    25
1    22
2    30
Name: Age, dtype: int64
```

---

### 🔹 ۴. دسترسی به ردیف‌ها

```python
print(df.loc[0])   # دسترسی با ایندکس برچسبی
print(df.iloc[1])  # دسترسی با ایندکس عددی
```

📌 خروجی:

```
Name     Ali
Age       25
Score     85
Name: 0, dtype: object

Name     Sara
Age        22
Score      90
Name: 1, dtype: object
```

---

### 🔹 ۵. فیلتر کردن داده‌ها

```python
print(df[df["Score"] > 80])   # فقط دانشجویانی که نمره بالای 80 دارند
```

📌 خروجی:

```
   Name  Age  Score
0   Ali   25     85
1  Sara   22     90
```

---

### 🔹 ۶. اضافه کردن ستون جدید

```python
df["Passed"] = df["Score"] >= 80
print(df)
```

📌 خروجی:

```
   Name  Age  Score  Passed
0   Ali   25     85    True
1  Sara   22     90    True
2  Reza   30     78   False
```

---

### 🔹 ۷. عملیات آماری

```python
print(df.describe())   # آمار کلی
print(df["Score"].mean())  # میانگین نمره‌ها
```

📌 خروجی:

```
             Age      Score
count   3.000000   3.000000
mean   25.666667  84.333333
std     4.041452   6.110101
min    22.000000  78.000000
25%    23.500000  81.500000
50%    25.000000  85.000000
75%    27.500000  87.500000
max    30.000000  90.000000

84.33333333333333
```

