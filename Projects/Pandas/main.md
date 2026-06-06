
# 📖 فصل حل تمرین: تحلیل داده‌های فروش آیفون  
✍️ نویسنده: سیامک عباس‌نژاد  

---------------------------------------------

## 🔹 ۱. معرفی فصل  

---------------------------------------------

## 🔹 ۲. بارگذاری کتابخانه‌ها و ساخت DataFrame اولیه

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
```

سپس تعریف داده‌ها:

```python

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

df = pd.DataFrame(data)
```

---------------------------------------------

## 🔹 ۳. پاک‌سازی داده‌ها (Data Cleaning)

### ۳.۱ حذف ردیف‌های دارای داده گمشده
```python
df.dropna(inplace=True)
df.dropna(inplace=True, subset=["Sale Price"])
```

### ۳.۲ پر کردن مقادیر گمشده با مقدار ثابت  
```python
print(df.fillna(999))
print(df["Mrp"].fillna(22222))
```

### ۳.۳ پر کردن با **میانگین MRP**
```python
print(df["Mrp"].mean())
print(df["Mrp"].fillna(df["Mrp"].mean()))
```

### ۳.۴ پر کردن **میانه‌ی قیمت فروش**
```python
print(df["Sale Price"].median())
print(df["Sale Price"].fillna(df["Sale Price"].median()))
```

### ۳.5 پر کردن مقادیر گمشده با **مد (mode)**
```python
x = df["Mrp"].mode()[0]
print(df["Mrp"].fillna(x))
```

---------------------------------------------

## 🔹 ۴. تبدیل داده‌ها (Data Transformation)

### ۴.۱ تبدیل تاریخ فروش به نوع datetime
```python
df["Sale Date"]= pd.to_datetime(df["Sale Date"])
print(df["Sale Date"])
```

### ۴.۲ تغییر مقدار یک سلول  
```python
df.loc[1, "Mrp"] = 69999
print(df["Mrp"])
```

### ۴.۳ حذف ردیف‌هایی که MRP کمتر از 25000 داشتند  
```python
for i in df.index:
    if df.loc[i, "Mrp"] == 25000:
        df.drop(i, inplace=True)
```

---------------------------------------------

## 🔹 ۵. بررسی و حذف داده‌های تکراری

```python
print(df.duplicated())
print(df.drop_duplicates())
```

---------------------------------------------

## 🔹 ۶. محاسبه همبستگی بین متغیرها

```python
print(df[["Mrp", "Sale Price", "Sale Date"]].corr)
```

---------------------------------------------

## 🔹 ۷. مدیریت ستون‌ها

### ۷.۱ افزودن ستون جدید
```python
df["col"] = 5
```

### ۷.۲ حذف ستون MRP
```python
df.drop(["Mrp"], axis=1, inplace=True)
print(df)
```

### ۷.۳ ادغام DataFrame جدید با اصلی  
```python
x = pd.DataFrame({
    "Mrp": 44444,
    "Sale Date": "1360/04/11"
}, index=[9])
y = pd.concat([x, df])
print(y)
```

### ۷.۴ افزودن یک رکورد جدید
```python
df.loc[len(df)+1, ["Number Of Ratings","Mrp", "Sale Price"]] = [55555, 888888, 2]
df.drop(0, inplace=True)
print(df)
```

---------------------------------------------

## 🔹 ۸. گروه‌بندی داده‌ها (Grouping)

### ۸.۱ میانگین قیمت فروش هر محصول  
```python
x = df.groupby("Product Name")["Sale Price"].mean()
```

### ۸.۲ استخراج مدل از نام محصول  
```python
df['Model'] = df['Product Name'].str.extract(r'(iPhone \d+ \w+)')
```

### ۸.۳ میانگین قیمت و تعداد نظرات برای هر مدل  
```python
result = df.groupby('Model').agg(
    Average_Sale_Price=('Sale Price', 'mean'),
    Average_Number_Of_Ratings=('Number Of Ratings', 'mean')
).reset_index()
print(result)
```

### ۸.۴ پیدا کردن گران‌ترین و ارزان‌ترین مدل  
```python
max_price_model = average_prices.loc[average_prices['Sale Price'].idxmax()]
min_price_model = average_prices.loc[average_prices['Sale Price'].idxmin()]
```

---------------------------------------------

## 🔹 ۹. مصورسازی داده‌ها

### ۹.۱ نمودار پراکندگی تعداد نظرات و قیمت  
```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Number Of Ratings', y='Sale Price')
plt.title('تعداد نظرات در برابر قیمت فروش')
plt.xlabel('تعداد نظرات')
plt.ylabel('قیمت فروش')
plt.show()
```

### ۹.۲ روند قیمت فروش در طول زمان  
```python
daily_average = df.groupby('Sale Date')['Sale Price'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(daily_average['Sale Date'], daily_average['Sale Price'], marker='o')
plt.title('روند میانگین قیمت فروش بر اساس تاریخ')
plt.xlabel('تاریخ')
plt.ylabel('میانگین قیمت فروش')
plt.grid()
plt.show()
```

---------------------------------------------

## 🔹 ۱۰. تحلیل قیمت‌های غیرمعمول (Outlier Detection)

```python
mean_sale_price = df['Sale Price'].mean()
unusual_price_threshold = 1.5 * mean_sale_price
unusual_products = df[df['Sale Price'] > unusual_price_threshold]

print(unusual_products[['Product Name', 'Sale Price']])
```

---------------------------------------------

## 🔹 ۱۱. افزودن ستون جدید در موقعیت دلخواه

```python
data = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
        {'a': 100, 'b': 200, 'c': 300, 'd': 400},
        {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000}]
df = pd.DataFrame(data)
df.insert(1, "h", [22, 88,99])
```

---------------------------------------------

## 🔹 ۱۲. محاسبه ستون جدید با apply

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})
df['Sum'] = df.apply(lambda row: row['A'] + row['B'], axis=1)
print(df)
```

---------------------------------------------

## 🔹۱۳. پیدا کردن ردیف‌هایی که شامل مقادیر Null هستند

**جواب:**  
می‌توانیم با استفاده از توابع `isnull` ،`isna` و دستورات شرطی، ایندکس ردیف‌هایی که شامل مقادیر Null هستند را پیدا کنیم.  

```python
import pandas as pd

# حلقه برای پیدا کردن ایندکس‌هایی که تمام مقادیرشان در آن ردیف null است
for i in df.index:
    # بررسی اینکه آیا هر مقدار در این ردیف null است یا خیر
    if pd.isnull(df.loc[i, df.columns]).any():
        print(i)

# پیدا کردن ایندکس‌هایی که مقدار ستون 'Sale Price' برابر NaN است
# nan_sale_price_indices = df[df['Sale Price'].isna()].index.tolist()
# print("\nIndices where 'Sale Price' is NaN:", nan_sale_price_indices)

# پیدا کردن ردیف‌هایی که در هر یک از ستون‌های مشخص شده مقدار NaN دارند
# mask = df[['Sale Price', 'Mrp', 'Sale Date']].isna().any(axis=1)
# indices_with_nans = df[mask].index.tolist()

# print("Indices with NaN values in selected columns:", indices_with_nans)
```
## 🔹 ۱۴. مدل‌سازی با رگرسیون جنگل تصادفی (Random Forest)

### ۱۴.۱ آماده‌سازی داده‌ها

```python
X = df[['Number Of Ratings']]
y = df['Sale Price']
```

### ۱۴.۲ تقسیم داده‌ها  

```python
X_train, X_test, y_train, y_test = train_test_split(...)
```


### ۱۴.۳ آموزش مدل  

```python
model = RandomForestRegressor()
model.fit(X_train, y_train)
```


### ۱۴.۴ پیش‌بینی و ارزیابی  

```python
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'میانگین مربعات خطا: {mse:.2f}')
```

