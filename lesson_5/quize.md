

### ❓ سؤال: حذف ردیف‌های با داده‌های گمشده و ذخیره تغییرات

```python
# تحلیل داده‌های فروش آیفون
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



data = {
    "Product Name": [
        "APPLE iPhone 8 Plus (Gold, 64 GB)",
        "APPLE iPhone 8 Plus (Space Grey, 256 GB)",
        "APPLE iPhone 8 Plus (Space Grey, 256 GB)",
        "APPLE iPhone 8 (Silver, 256 GB)",
        "APPLE iPhone 8 (Silver, 256 GB)",
        "APPLE iPhone 8 Plus (Silver, 64 GB)",
        "APPLE iPhone 8 Plus (Space Grey, 64 GB)",
        "APPLE iPhone 8 (Space Grey, 256 GB)",
        "APPLE iPhone XS Max (Silver, 64 GB)"
    ],
    "Sale Price": [
        49900.0,
        84900.0,
        84900.0,
        77000.0,
        77000.0,
        None,
        49900.0,
        77000.0,
        89900.0
    ],
    "Mrp": [
        49900.0,
        1.0,
        84900.0,
        77000.0,
        77000.0,
        49900.0,
        None,
        77000.0,
        89900.0
    ],
    "Number Of Ratings": [
        3431,
        3431,
        3431,
        11202,
        11202,
        3431,
        3431,
        11202,
        1454
    ],
    "Sale Date": [
        "14/1/2023",
        "15/1/2023",
        "16/1/2023",
        "17/1/2023",
        "17/1/2023",
        "19/1/2023",
        "20/1/2023",
        None,
        "22/1/2023"
    ]
}
# Remove rows with any missing values
df.dropna(inplace=True)
```

---

### ❓ سؤال: حذف ردیف‌های با قیمت فروش گمشده

```python
# Remove rows where 'Sale Price' is missing
df.dropna(inplace=True, subset=["Sale Price"])
```

---

### ❓ سؤال: پر کردن مقادیر گمشده با عدد 999 و چاپ DataFrame

```python
# Fill missing values with 999 and print the DataFrame
print(df.fillna(999))
```

---

### ❓ سؤال: پر کردن مقادیر گمشده در ستون MRP با عدد 22222 و چاپ

```python
# Fill missing values in 'Mrp' column with 22222 and print it
print(df["Mrp"].fillna(22222))
```

---

### ❓ سؤال: محاسبه میانگین MRP و چاپ آن

```python
# Calculate the mean of the 'Mrp' column and print it
print(df["Mrp"].mean())
```

---

### ❓ سؤال: پر کردن مقادیر گمشده در ستون MRP با میانگین MRP و چاپ

```python
# Fill missing 'Mrp' values with the column mean and print
print(df["Mrp"].fillna(df["Mrp"].mean()))
```

---

### ❓ سؤال: محاسبه میانه قیمت فروش و چاپ آن

```python
# Calculate and print the median of the 'Sale Price' column
print(df["Sale Price"].median())
```

---

### ❓ سؤال: پر کردن مقادیر گمشده در ستون قیمت فروش با میانه قیمت فروش و چاپ

```python
# Fill missing 'Sale Price' values with the column median and print
print(df["Sale Price"].fillna(df["Sale Price"].median()))
```

---

### ❓ سؤال: محاسبه مد (mode) برای ستون MRP و پر کردن مقادیر گمشده با آن

```python
# Fill missing 'Mrp' values with the mode (most frequent value) and print
x = df["Mrp"].mode()[0]
print(df["Mrp"].fillna(x))
```

---

### ❓ سؤال: تبدیل تاریخ فروش به فرمت تاریخ و چاپ آن

```python
# Convert 'Sale Date' to datetime format and print
df["Sale Date"] = pd.to_datetime(df["Sale Date"])
print(df["Sale Date"])
```

---

### ❓ سؤال: تغییر مقدار MRP در ردیف دوم به 69999 و چاپ

```python
# Change the 'Mrp' value in row 1 to 69999 and print
df.loc[1, "Mrp"] = 69999
print(df["Mrp"])
```

---

### ❓ سؤال: تنظیم حداقل مقدار MRP به 25000 برای ردیف‌های موجود

```python
# Set the minimum 'Mrp' value to 25000 for rows where it is 25000
for i in df.index:
    if df.loc[i, "Mrp"] == 25000:
        df.loc[i, "Mrp"] = 25000
print(df["Mrp"])
```

---

### ❓ سؤال: حذف ردیف‌هایی که MRP آن‌ها کمتر از 25000 است

```python
# Remove rows with 'Mrp' equal to 25000
for i in df.index:
    if df.loc[i, "Mrp"] == 25000:
        df.drop(i, inplace=True)
print(df["Mrp"])
```

---

### ❓ سؤال: بررسی وجود ردیف‌های تکراری و چاپ نتیجه

```python
# Check for duplicate rows and print result
print(df.duplicated())
```

---

### ❓ سؤال: حذف ردیف‌های تکراری و چاپ DataFrame

```python
# Drop duplicate rows and print the DataFrame
print(df.drop_duplicates())
```

---

### ❓ سؤال: محاسبه همبستگی بین ستون‌های MRP، قیمت فروش و تاریخ فروش و چاپ

```python
# Calculate correlation between 'Mrp', 'Sale Price', and 'Sale Date'
print(df[["Mrp", "Sale Price", "Sale Date"]].corr)
```

---

### ❓ سؤال: اضافه کردن یک ستون جدید با مقدار ثابت 5

```python
# Add a new column named 'col' with constant value 5
df["col"] = 5
```

---

### ❓ سؤال: حذف ستون‌های MRP و تاریخ فروش از DataFrame

```python
# Drop 'Mrp' and 'Sale Date' columns from the DataFrame
df.drop(["Mrp"], axis=1, inplace=True)
print(df)
```

---

### ❓ سؤال: ایجاد یک DataFrame جدید با MRP و تاریخ فروش

```python
# Create a new DataFrame with 'Mrp' and 'Sale Date'
x = pd.DataFrame({
    "Mrp": 44444,
    "Sale Date": "1360/04/11"
}, index=[9])
```

---

### ❓ سؤال: ادغام DataFrame جدید با DataFrame اصلی

```python
# Concatenate the new DataFrame with the original one
y = pd.concat([x, df])
print(y)
```

---

### ❓ سؤال: اضافه کردن یک ردیف جدید به DataFrame با مشخصات تعداد نظرات، MRP و قیمت فروش

```python
# Add a new row with 'Number Of Ratings', 'Mrp', and 'Sale Price'
df.loc[len(df)+1, ["Number Of Ratings", "Mrp", "Sale Price"]] = [55555, 888888, 2]
```

---

### ❓ سؤال: حذف ردیف اول از DataFrame

```python
# Drop the first row of the DataFrame
df.drop(0, inplace=True)
print(df)
```

