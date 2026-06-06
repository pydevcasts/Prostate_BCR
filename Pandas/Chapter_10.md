# 📖 فصل ۱۰: داده‌های زمانی (Time Series)

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. چرا داده‌های زمانی مهم هستند؟

بسیاری از داده‌های دنیای واقعی مثل:

* قیمت سهام در روزهای مختلف
* دمای هوا در طول سال
* فروش ماهانه یک شرکت

به صورت **سری زمانی (Time Series)** ذخیره می‌شوند.
Pandas ابزارهای قدرتمندی برای کار با داده‌های زمانی دارد.

---

### 🔹 ۲. ساخت Series زمانی

```python
import pandas as pd

# Create a date range
dates = pd.date_range("2023-01-01", periods=6, freq="D")

# Create a time series
data = pd.Series([10, 20, 15, 30, 25, 40], index=dates)
print(data)
```

📌 خروجی:

```
2023-01-01    10
2023-01-02    20
2023-01-03    15
2023-01-04    30
2023-01-05    25
2023-01-06    40
Freq: D, dtype: int64
```

---

### 🔹 ۳. انتخاب داده‌های خاص

```python
# Select by date
print(data["2023-01-03"])

# Select by range
print(data["2023-01-02":"2023-01-05"])
```

---

### 🔹 ۴. تغییر فرکانس (Resampling)

```python
# Change daily data to weekly (sum)
print(data.resample("W").sum())

# Change daily data to monthly (mean)
print(data.resample("M").mean())
```

---

### 🔹 ۵. کار با ستون‌های زمانی در DataFrame

```python
df = pd.DataFrame({
    "Date": pd.date_range("2023-01-01", periods=5, freq="D"),
    "Sales": [100, 150, 200, 250, 300]
})

# Convert column to datetime type
df["Date"] = pd.to_datetime(df["Date"])

# Extract year, month, day
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

print(df)
```

📌 خروجی:

```
        Date  Sales  Year  Month  Day
0 2023-01-01    100  2023      1    1
1 2023-01-02    150  2023      1    2
2 2023-01-03    200  2023      1    3
3 2023-01-04    250  2023      1    4
4 2023-01-05    300  2023      1    5
```

---

### 🔹 ۶. جابجایی و محاسبات با داده‌های زمانی

```python
# Shift data forward by 1 day
print(data.shift(1))

# Calculate daily change
print(data.diff())
```

📌 خروجی:

```
2023-01-01     NaN
2023-01-02    10.0
2023-01-03    -5.0
2023-01-04    15.0
2023-01-05    -5.0
2023-01-06    15.0
Freq: D, dtype: float64
```

---
### 🔹۷.   اضافه کردن ستون سن کاربر و مدت زمان گذشته از آخرین خرید
```python
import pandas as pd
from datetime import datetime

data = pd.DataFrame({
    'name': ['Ali', 'Sara', 'Reza'],
    'birth_date': ['1998-07-15', '1985-01-12', '2002-05-03'],
    'purchase_amount': [320000, 920000, 180000],
    'last_purchase': ['2024-11-10', '2024-11-05', '2024-10-22']
})

df = pd.DataFrame(data)
df["birth_date"] = pd.to_datetime(df["birth_date"])
df['last_purchase'] = pd.to_datetime(data['last_purchase'])
df.info()

today = datetime(2026,2,2)
df["age"] = (today - df['birth_date']).dt.days // 365

# 🕓 ساخت ویژگی فاصله از آخرین خرید
df['days_since_last_purchase'] = (today - df['last_purchase']).dt.days
df
```

### 🔹 ۷. تمرین پیشنهادی

یک DataFrame بسازید که شامل:

* تاریخ‌های یک ماه
* دمای هوا در هر روز

سپس:

1. میانگین دما در هر هفته را حساب کنید.
2. اختلاف دمای هر روز با روز قبل را پیدا کنید.
3. بیشترین دما در کل ماه را نمایش دهید.

