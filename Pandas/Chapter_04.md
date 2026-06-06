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

### 🔹 ۵. چهار عمل اصلی داده در SQL Database

اگر داده‌های شما در دیتابیس ذخیره شده باشد، می‌توانید مستقیم آن را با Pandas بخوانید.

```python
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect("data.db")
c = conn.cursor()

# c.execute('''
#     CREATE TABLE IF NOT EXISTS hotel (
#         FIND INTEGER PRIMARY KEY NOT NULL,
#         FNAME TEXT NOT NULL,
#         COST INTEGER NOT NULL,
#         WEIGHT INTEGER,
#         PASSWORD TEXT NOT NULL
#     )
# ''')

# def add_hotel(id, fname, cost, weight, password):
#     hashed = hashlib.sha256(password.encode('utf-8')).hexdigest()
#     c.execute("INSERT INTO hotel (FIND, FNAME, COST, WEIGHT, PASSWORD) VALUES (?,?,?,?,?)", (id, fname, cost, weight, password))
# add_hotel(1,"Maryam", 24, 58,"123")
# برای زمانی که پسورد نداشته باشیم

# c.execute("INSERT INTO hotel (FIND, FNAME, COST, WEIGHT) VALUES (3, 'iceream', 200, 30)")
# print(x.fetchall())

# x = c.execute("SELECT FNAME FROM hotel WHERE FNAME='Cakes'")
# x = c.execute("UPDATE hotel SET FNAME='Icecream' WHERE FIND=3")

# DROP FROM hotel  emit table
# c.execute("DELETE FROM hotel where FIND > 2")
# c.execute("ALTER TABLE hotel DROP COLUMN FNAME")
conn.commit()
conn.close()

# خواندن جدول به صورت DataFrame
df_sql = pd.read_sql_query("SELECT * FROM hotel", conn)
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
