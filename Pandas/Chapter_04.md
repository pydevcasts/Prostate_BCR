# 📖 فصل ۴: خواندن و نوشتن فایل‌ها در Pandas

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. خواندن فایل CSV

بیشتر داده‌هایی که در علم داده استفاده می‌کنیم در قالب **CSV (Comma Separated Values)** ذخیره می‌شوند.

```python
import pandas as pd

# خواندن فایل csv
df = pd.read_csv("data.csv")

print(df.head())   # نمایش ۵ ردیف اول
```

📌 متد `head()` برای نمایش چند ردیف اول بسیار کاربردی است.

---

### 🔹 ۲. نوشتن DataFrame در فایل CSV

```python
df.to_csv("output.csv", index=False)
```

📌 پارامتر `index=False` یعنی ایندکس‌ها در فایل ذخیره نشوند.

---

### 🔹 ۳. خواندن فایل Excel

```python
df_excel = pd.read_excel("data.xlsx", sheet_name="Sheet1")
print(df_excel.head())
```

📌 برای کار با فایل‌های اکسل نیاز دارید بسته‌ی **openpyxl** را نصب کنید:

```bash
pip install openpyxl
```

---

### 🔹 ۴. نوشتن DataFrame در Excel

```python
df.to_excel("output.xlsx", index=False, sheet_name="MyData")
```

---

### 🔹 ۵. خواندن داده از SQL Database

اگر داده‌های شما در دیتابیس ذخیره شده باشد، می‌توانید مستقیم آن را با Pandas بخوانید.

```python
import sqlite3

# اتصال به دیتابیس
conn = sqlite3.connect("my_database.db")

# خواندن جدول به صورت DataFrame
df_sql = pd.read_sql_query("SELECT * FROM students", conn)

print(df_sql.head())
```

---

### 🔹 ۶. نکته‌های مهم

* متد `read_csv()` می‌تواند پارامترهای زیادی داشته باشد (مثل جداکننده‌ی ستون‌ها، نام ستون‌ها و …).
* همیشه بعد از خواندن داده، بهتر است یک `df.info()` بزنید تا وضعیت کلی داده‌ها مشخص شود.

---

### 🔹 ۷. بررسی اطلاعات اولیه داده‌ها

```python
print(df.info())      # نمایش نوع داده و تعداد مقادیر
print(df.describe())  # آمار توصیفی
print(df.columns)     # نام ستون‌ها
print(df.shape)       # تعداد ردیف و ستون
```
