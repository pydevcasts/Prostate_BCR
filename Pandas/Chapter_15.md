# 📖 فصل ۱۵: ورودی و خروجی داده‌ها در Pandas (I/O Operations)

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. مقدمه

یکی از مهم‌ترین کاربردهای Pandas، خواندن و ذخیره‌سازی داده‌هاست.
فرمت‌های پرکاربرد شامل:

* **CSV**
* **Excel**
* **SQL Database**
* **JSON**

در این فصل یاد می‌گیریم چطور با این فرمت‌ها کار کنیم.

---

### 🔹 ۲. کار با فایل‌های CSV

```python
import pandas as pd

# Read CSV file
df = pd.read_csv("data.csv")
print(df.head())

# Save DataFrame to CSV
df.to_csv("output.csv", index=False)
```

📌 نکته:
گزینه `index=False` باعث میشه اندیس به عنوان ستون اضافی ذخیره نشه.

---

### 🔹 ۳. کار با فایل‌های Excel

```python
# Read Excel file
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
print(df.head())

# Save DataFrame to Excel
df.to_excel("output.xlsx", sheet_name="Results", index=False)
```

📌 برای کار با Excel باید کتابخانه‌ی `openpyxl` یا `xlsxwriter` نصب باشه.

```bash
pip install openpyxl
```

---

### 🔹 ۴. کار با پایگاه داده SQL

```python
import sqlite3

# Connect to database (create if not exists)
conn = sqlite3.connect("mydata.db")

# Read table from SQL
df = pd.read_sql("SELECT * FROM sales", conn)

# Write DataFrame to SQL
df.to_sql("new_sales", conn, if_exists="replace", index=False)

conn.close()
```

---

### 🔹 ۵. کار با فایل‌های JSON

```python
# Read JSON file
df = pd.read_json("data.json")
print(df.head())

# Save DataFrame to JSON
df.to_json("output.json", orient="records", lines=True)
```

📌 گزینه `orient` نحوه نمایش JSON رو مشخص می‌کنه.

---

### 🔹 ۶. خواندن بخشی از فایل (برای فایل‌های بزرگ)

```python
# Read first 100 rows only
df = pd.read_csv("bigdata.csv", nrows=100)

# Read file in chunks
chunks = pd.read_csv("bigdata.csv", chunksize=500)
for chunk in chunks:
    print(chunk.shape)
```

---

### 🔹 ۷. تمرین پیشنهادی

۱. یک دیتافریم بسازید و آن را در فرمت‌های **CSV**، **Excel** و **JSON** ذخیره کنید.
۲. دوباره فایل‌ها را بخوانید و بررسی کنید آیا داده‌ها به درستی ذخیره شده‌اند.
3\. یک دیتابیس SQLite ایجاد کنید، دیتافریم را در آن ذخیره کنید و سپس دوباره داده‌ها را بخوانید.

