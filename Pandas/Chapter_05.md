# 📖 فصل ۵: پاک‌سازی داده‌ها (Data Cleaning)

✍️ نویسنده: سیامک عباس‌نژاد

---

یکی از مهم‌ترین مراحل در **علم داده (Data Science)**، مرحله‌ی **پاک‌سازی داده‌ها** است.
چون معمولاً داده‌ها:

* ناقص (Missing Values)
* تکراری (Duplicates)
* یا دارای داده‌های نامعتبر هستند.

Pandas ابزارهای قدرتمندی برای پاک‌سازی داده‌ها دارد.

---

### 🔹 ۱. بررسی داده‌های گم‌شده (Missing Values)

```python
import pandas as pd

data = {
    "Name": ["Ali", "Sara", "Reza", "Niloofar"],
    "Age": [25, None, 30, 28],
    "Score": [85, 90, None, 95]
}

df = pd.DataFrame(data)
print(df)
```

📌 خروجی:

```
       Name   Age  Score
0       Ali  25.0   85.0
1      Sara   NaN   90.0
2      Reza  30.0    NaN
3  Niloofar  28.0   95.0
```

---

### 🔹 ۲. پیدا کردن مقادیر گم‌شده

```python
print(df.isnull())       # نمایش True/False برای هر سلول
print(df.isnull().sum()) # تعداد داده‌های گم‌شده در هر ستون
```

📌 خروجی:

```
    Name    Age  Score
0  False  False  False
1  False   True  False
2  False  False   True
3  False  False  False

Name     0
Age      1
Score    1
dtype: int64
```

---

### 🔹 ۳. پر کردن داده‌های گم‌شده

```python
df["Age"].fillna(df["Age"].mean(), inplace=True)     # پر کردن با میانگین
df["Score"].fillna(df["Score"].median(), inplace=True)  # پر کردن با میانه
```

📌 حالا ستون‌ها کامل می‌شوند.

---

### 🔹 ۴. حذف داده‌های گم‌شده

```python
df_drop = df.dropna()
print(df_drop)
```

📌 این دستور ردیف‌هایی که مقدار گم‌شده دارند را حذف می‌کند.

---

### 🔹 ۵. حذف داده‌های تکراری

```python
df = pd.DataFrame({
    "Name": ["Ali", "Sara", "Sara", "Reza"],
    "Age": [25, 22, 22, 30]
})

print(df.duplicated())    # بررسی تکراری بودن
print(df.drop_duplicates())  # حذف ردیف‌های تکراری
```

---

### 🔹 ۶. تغییر نوع داده‌ها (Data Types)

```python
df["Age"] = df["Age"].astype(int)
print(df.dtypes)
```

📌 این کار زمانی مهم است که مثلاً داده‌های عددی به صورت رشته (string) وارد شده باشند.

---

### 🔹 ۷. تغییر نام ستون‌ها

```python
df.rename(columns={"Name": "StudentName", "Age": "StudentAge"}, inplace=True)
print(df.head())
```
