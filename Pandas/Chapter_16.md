# 📖 فصل ۱۶: بهینه‌سازی عملکرد و مدیریت حافظه در Pandas

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. مقدمه

وقتی با داده‌های کوچک کار می‌کنیم، Pandas خیلی سریع و روان اجرا می‌شود.
اما در پروژه‌های واقعی (مثل داده‌های مالی، شبکه‌های اجتماعی، یا لاگ‌های سیستمی) حجم داده‌ها ممکن است به چندین **گیگابایت** برسد.
اگر اصول مدیریت حافظه و بهینه‌سازی را ندانیم، ممکن است:

* اجرای کد خیلی کند شود
* رم (RAM) پر شود و کرش کند

در این فصل نکاتی برای بهینه‌سازی کار با Pandas یاد می‌گیریم.

---

### 🔹 ۲. بررسی میزان استفاده از حافظه

```python
import pandas as pd

# Load sample data
df = pd.DataFrame({
    "A": range(1, 10001),
    "B": ["Category"] * 10000,
    "C": [3.14] * 10000
})

# Check memory usage
print(df.info(memory_usage="deep"))
```

📌 خروجی نشان می‌دهد هر ستون چه مقدار حافظه استفاده می‌کند.

---

### 🔹 ۳. تبدیل نوع داده‌ها (Downcasting)

```python
# Before optimization
print(df["A"].dtype)

# Convert int64 → int32
df["A"] = pd.to_numeric(df["A"], downcast="integer")

# Convert float64 → float32
df["C"] = pd.to_numeric(df["C"], downcast="float")

print(df.dtypes)
```

📌 این کار باعث کاهش مصرف حافظه می‌شود بدون اینکه دقت محاسبات تغییر کند.

---

### 🔹 ۴. استفاده از نوع داده دسته‌ای (Categorical)

```python
# Convert object to category
df["B"] = df["B"].astype("category")

print(df.dtypes)
```

📌 ستون‌های متنی که مقادیر تکراری دارند (مثل شهرها یا دسته‌بندی‌ها) بهتر است به `category` تبدیل شوند.

---

### 🔹 ۵. خواندن فایل‌های بزرگ به صورت تکه‌ای (Chunking)

```python
# Read large file in chunks
chunks = pd.read_csv("bigdata.csv", chunksize=100000)

for chunk in chunks:
    print(chunk.shape)  # process each chunk separately
```

📌 به جای بارگذاری کل فایل در حافظه، آن را بخش‌بخش پردازش می‌کنیم.

---

### 🔹 ۶. انتخاب ستون‌های مورد نیاز

```python
# Read only specific columns
df = pd.read_csv("bigdata.csv", usecols=["Name", "Age", "Salary"])
```

📌 همیشه فقط ستون‌هایی که لازم داریم بخوانیم، نه کل جدول.

---

### 🔹 ۷. تمرین پیشنهادی

۱. یک دیتافریم شامل ۱ میلیون ردیف و چند ستون بسازید.
۲. بررسی کنید هر ستون چه مقدار حافظه استفاده می‌کند.
۳. ستون‌های عددی را با `downcast` و ستون‌های متنی را با `category` بهینه کنید.
۴. مصرف حافظه قبل و بعد را مقایسه کنید.
