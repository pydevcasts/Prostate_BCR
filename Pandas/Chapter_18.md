# 📖 ضمیمه: تقلب‌نامه (Cheat Sheet) Pandas

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. شروع کار با Pandas

```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

# Read data
df = pd.read_csv("file.csv")
df = pd.read_excel("file.xlsx")

# Save data
df.to_csv("output.csv", index=False)
df.to_excel("output.xlsx", index=False)
```

---

### 🔹 ۲. بررسی داده‌ها

```python
df.head()          # نمایش اولین ردیف‌ها
df.tail()          # نمایش آخرین ردیف‌ها
df.info()          # خلاصه اطلاعات
df.describe()      # آمار توصیفی
df.shape           # ابعاد دیتافریم
df.columns         # نام ستون‌ها
df.dtypes          # نوع داده هر ستون
```

---

### 🔹 ۳. انتخاب داده‌ها

```python
df["A"]                # انتخاب یک ستون
df[["A", "B"]]         # انتخاب چند ستون
df.iloc[0]             # انتخاب ردیف اول
df.iloc[0:5]           # انتخاب ۵ ردیف اول
df.loc[0, "A"]         # انتخاب مقدار مشخص
```

---

### 🔹 ۴. فیلترگذاری

```python
df[df["A"] > 10]           # شرط روی ستون
df[(df["A"] > 10) & (df["B"] < 5)]  # چند شرط
df.query("A > 10 and B < 5")        # شرط با query
```

---

### 🔹 ۵. مدیریت داده‌های گمشده

```python
df.isnull().sum()          # شمارش مقادیر Null
df.dropna()                # حذف ردیف‌های Null
df.fillna(0)               # جایگزینی Null با مقدار
df["A"].fillna(df["A"].mean(), inplace=True)  # میانگین
```

---

### 🔹 ۶. عملیات روی ستون‌ها

```python
df["C"] = df["A"] + df["B"]        # ساخت ستون جدید
df["C"] = df["A"] * 2              # تغییر مقدار ستون
df.rename(columns={"A": "Age"}, inplace=True)  # تغییر نام
df.drop(columns=["B"], inplace=True)           # حذف ستون
```

---

### 🔹 ۷. گروه‌بندی و آمار

```python
df.groupby("Category")["Value"].mean()
df.groupby(["A", "B"]).sum()
df.pivot_table(values="Value", index="A", columns="B", aggfunc="mean")
pd.crosstab(df["A"], df["B"])
```

---

### 🔹 ۸. ادغام و اتصال داده‌ها

```python
pd.concat([df1, df2], axis=0)           # اتصال عمودی
pd.concat([df1, df2], axis=1)           # اتصال افقی
df1.merge(df2, on="ID", how="inner")    # merge (inner, left, right, outer)
df1.join(df2, lsuffix="_x", rsuffix="_y")
```

---

### 🔹 ۹. مرتب‌سازی

```python
df.sort_values("A", ascending=True)      # مرتب‌سازی بر اساس ستون
df.sort_index()                          # مرتب‌سازی بر اساس اندیس
```

---

### 🔹 ۱۰. بهینه‌سازی حافظه

```python
df["A"] = pd.to_numeric(df["A"], downcast="integer")  
df["Category"] = df["Category"].astype("category")  
```

---

