# 📖 فصل ۵: پاک‌سازی داده‌ها (Data Cleaning)

✍️ نویسنده: سیامک عباس‌نژاد

---

یکی از مهم‌ترین مراحل در **علم داده (Data Science)**، مرحله‌ی **پاک‌سازی داده‌ها** است.
چون معمولاً داده‌ها:

* ناقص (Missing Values)
* تکراری (Duplicates)
* یا دارای داده‌های نامعتبر هستند.

کتابخانه‌ی قدرتمند pandas ابزارهای بسیار مفیدی برای پاک‌سازی داده‌ها در اختیار ما قرار می‌دهد.

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
print(df.isnull())        # نمایش True/False برای هر سلول
print(df.isnull().sum())  # تعداد داده‌های گم‌شده در هر ستون
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

📌 بررسی اینکه ستون "Name" نال دارد یا نه:

```python
df["Name"].isnull().any()
```

📌 تعداد Null های ستون "Name":

```python
df["Name"].isnull().sum()
```

📌 نمایش فقط ستون‌هایی که حداقل یک مقدار Null دارند:

```python
df.columns[df.isnull().any()]
```

---

📌 «فقط مقادیر ستون Name را در ردیف‌هایی که دارای Null هستند نمایش دهیم»

```python
df.loc[df.isnull().any(axis=1), "Name"]
```

---
### 🔹 ۳. پر کردن داده‌های گم‌شده

```python
df["Age"].fillna(df["Age"].mean(), inplace=True)     # پر کردن با میانگین
df["Score"] = df["Score"].fillna(df["Score"].median(), inplace=True)  # پر کردن با میانه
```

📌 حالا ستون‌ها کامل می‌شوند و دیگر مقدار NaN نخواهند داشت.

---

### 🔹 ۴. حذف داده‌های گم‌شده

📌 حذف تمام ردیف‌هایی که حداقل یک مقدار گم‌شده دارند:

```python
df_drop = df.dropna()
print(df_drop)
```

📌 حذف سطرهایی که در ستون خاصی Null دارند:

```python
df = df.dropna(subset=['Age'])
```

📌 حذف سطرهایی که همه مقادیرشان Null است:

```python
df = df.dropna(how='all')
```

---

### 🔹 ۵. حذف سطرها با دستور Drop

📌 حذف بر اساس ایندکس:

```python
df = df.drop(3)  # حذف سطر با ایندکس 3
```

📌 حذف چند سطر همزمان:

```python
df = df.drop([1, 4, 7])
```

📌 حذف در همان دیتافریم (بدون نیاز به انتساب مجدد):

```python
df.drop(3, inplace=True)
```

---

### 🔹 ۶. حذف داده‌های تکراری

```python
df = pd.DataFrame({
    "Name": ["Ali", "Sara", "Sara", "Reza"],
    "Age": [25, 22, 22, 30]
})

print(df.duplicated())         # بررسی تکراری بودن
print(df.drop_duplicates())    # حذف ردیف‌های تکراری
```

---

### 🔹 ۷. تغییر نوع داده‌ها (Data Types)

```python
df["Age"] = df["Age"].astype(int)
print(df.dtypes)
```

📌 این کار زمانی مهم است که مثلاً داده‌های عددی به صورت رشته (string) وارد شده باشند.

📌 مشاهده خلاصه اطلاعات دیتافریم:

```python
df.info()
```

---

### 🔹 ۸. تغییر نام ستون‌ها

```python
df.rename(columns={"Name": "StudentName", "Age": "StudentAge"}, inplace=True)
print(df.head())
```
