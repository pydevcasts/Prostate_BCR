# آموزش جامع کتابخانه Pandas 📊

کتابخانه Pandas یکی از مهم‌ترین و پرکاربردترین کتابخانه‌های پایتون برای تحلیل داده‌ها و کار با داده‌های جدولی است. این کتابخانه به ویژه در زمینه‌های علم داده، تحلیل داده‌ها، و یادگیری ماشین بسیار مورد استفاده قرار می‌گیرد. Pandas به شما این امکان را می‌دهد که به راحتی داده‌ها را بارگذاری، پردازش، و تحلیل کنید. در این مقاله، به بررسی ویژگی‌ها و توابع اصلی این کتابخانه می‌پردازیم.

## 1. نصب Pandas 🛠️

برای نصب کتابخانه Pandas می‌توانید از pip استفاده کنید. در خط فرمان یا ترمینال خود، دستور زیر را اجرا کنید:

```bash
pip install pandas
```

اگر از Anaconda استفاده می‌کنید، می‌توانید Pandas را با استفاده از دستور زیر نصب کنید:

```bash
conda install pandas
```

## 2. وارد کردن Pandas 📥

پس از نصب Pandas، می‌توانید آن را در کد خود وارد کنید:

```python
import pandas as pd
```

## 3. ساخت DataFrame و Series 📋

### 3.1. ساخت Series

یک آرایه یک بعدی است که می‌تواند هر نوع داده‌ای را در خود جای دهد. به عنوان مثال:

```python
import pandas as pd

# Create a Series
data = [1, 2, 3, 4, 5]
series = pd.Series(data)
print(series)
```

### 3.2. ساخت DataFrame

یک ساختار داده دو بعدی است که می‌تواند داده‌های جدولی را ذخیره کند. به عنوان مثال:

```python
import pandas as pd

# Create a DataFrame
data = {
    'Name': ['Ali', 'Mehdi', 'Sara'],
    'Age': [25, 30, 22],
    'Job': ['Engineer', 'Doctor', 'Teacher']
}
df = pd.DataFrame(data)
print(df)
```

## 4. خواندن و نوشتن داده‌ها 📂

### 4.1. خواندن داده‌ها از فایل CSV

برای خواندن داده‌ها از یک فایل CSV، از تابع `read_csv()` استفاده می‌شود:

```python
df = pd.read_csv('data.csv')
print(df)
```

### 4.2. نوشتن داده‌ها به فایل CSV

برای نوشتن داده‌ها به یک فایل CSV، از تابع `to_csv()` استفاده می‌شود:

```python
df.to_csv('output.csv', index=False)
```

## 5. بررسی داده‌ها 🔍

### 5.1. نمایش اولین و آخرین سطرها

برای نمایش اولین چند سطر از DataFrame از تابع `head()` و برای نمایش آخرین چند سطر از `tail()` استفاده می‌شود:

```python
print(df.head())  # Display the first 5 rows
print(df.tail(3))  # Display the last 3 rows
```

### 5.2. اطلاعات کلی درباره DataFrame

برای دریافت اطلاعات کلی درباره DataFrame (شامل تعداد سطرها و ستون‌ها، نوع داده‌ها و ...) می‌توانید از تابع `info()` استفاده کنید:

```python
print(df.info())
```

### 5.3. آمار توصیفی

برای دریافت آمار توصیفی از داده‌ها، می‌توانید از تابع `describe()` استفاده کنید:

```python
print(df.describe())
```

## 6. انتخاب و فیلتر کردن داده‌ها 🗂️

### 6.1. انتخاب ستون‌ها

برای انتخاب یک یا چند ستون از DataFrame، می‌توانید از نام ستون‌ها استفاده کنید:

```python
# Select a single column
print(df['Name'])

# Select multiple columns
print(df[['Name', 'Age']])
```

### 6.2. فیلتر کردن سطرها

برای فیلتر کردن سطرها بر اساس شرایط خاص، می‌توانید از عبارات منطقی استفاده کنید:

```python
# Filter rows based on age
filtered_df = df[df['Age'] > 25]
print(filtered_df)
```

## 7. دستکاری داده‌ها ✏️

### 7.1. اضافه کردن ستون جدید

برای اضافه کردن یک ستون جدید به DataFrame، می‌توانید به سادگی نام ستون جدید را مشخص کنید:

```python
df['Income'] = [50000, 70000, 60000]
print(df)
```

### 7.2. حذف ستون

برای حذف یک ستون، می‌توانید از تابع `drop()` استفاده کنید:

```python
df = df.drop('Income', axis=1)
print(df)
```

### 7.3. تغییر نام ستون‌ها

برای تغییر نام ستون‌ها، می‌توانید از تابع `rename()` استفاده کنید:

```python
df = df.rename(columns={'Age': 'Age_Year'})
print(df)
```

## 8. گروه‌بندی داده‌ها 🥳

برای گروه‌بندی داده‌ها و انجام محاسبات روی گروه‌ها، می‌توانید از تابع `groupby()` استفاده کنید:

```python
grouped = df.groupby('Job').mean()
print(grouped)
```

## 9. پردازش داده‌های مفقود 🕵️

برای شناسایی داده‌های مفقود (NaN) در DataFrame، می‌توانید از تابع `isnull()` استفاده کنید:

```python
print(df.isnull().sum())  # Count of missing values in each column
```

برای حذف سطرها یا ستون‌های حاوی مقادیر مفقود، می‌توانید از تابع `dropna()` استفاده کنید:

```python
df = df.dropna()  # Drop rows with missing values
```

## 10. ادغام و اتصال داده‌ها 🔗

### 10.1. ادغام DataFrame ها

برای ادغام دو DataFrame با استفاده از یک کلید مشترک، می‌توانید از تابع `merge()` استفاده کنید:

```python
df1 = pd.DataFrame({
    'Key': ['A', 'B', 'C'],
    'Value1': [1, 2, 3]
})

df2 = pd.DataFrame({
    'Key': ['A', 'B', 'D'],
    'Value2': [4, 5, 6]
})

merged_df = pd.merge(df1, df2, on='Key', how='inner')
print(merged_df)
```

### 10.2. اتصال DataFrame ها

برای اتصال دو DataFrame به صورت عمودی یا افقی، می‌توانید از توابع `concat()` استفاده کنید:

```python
# Vertical concatenation
concat_df = pd.concat([df1, df2], axis=0)
print(concat_df)

# Horizontal concatenation
concat_df_h = pd.concat([df1, df2], axis=1)
print(concat_df_h)
```

## 11. ذخیره‌سازی داده‌ها 📥

برای ذخیره‌سازی داده‌ها به فرمت‌های مختلف، می‌توانید از توابع مختلفی مثل `to_excel()`, `to_json()` و `to_sql()` استفاده کنید:

```python
# Save to Excel file
df.to_excel('output.xlsx', index=False)

# Save to JSON file
df.to_json('output.json')
```

## 12. مثال کامل: تحلیل داده‌ها

در اینجا یک مثال کامل از بارگذاری داده‌ها، پردازش و تحلیل آن‌ها آورده شده است:

```python
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Display general information
print(df.info())

# Drop rows with missing values
df = df.dropna()

# Group by a column and calculate the mean
grouped = df.groupby('Job').mean()
print(grouped)

# Save results to a CSV file
grouped.to_csv('grouped_output.csv')
```



