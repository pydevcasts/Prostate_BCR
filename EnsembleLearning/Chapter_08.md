

## 🔹 بخش ۲: مهندسی ویژگی‌ها و حذف نقاط پرت

هدف این بخش آماده‌سازی داده‌ها برای ورود به مدل است.
اینجا داده‌های پرت حذف، ویژگی‌های ترکیبی ساخته، و متغیرهای متنی به عددی تبدیل می‌شوند.

---

### 🧹 ۱ — حذف نقاط پرت

```python
# Remove outliers based on domain thresholds
dataset.drop(dataset[dataset['CreditScore'] <= 359].index, inplace=True)
dataset.drop(dataset[dataset['Age'] >= 71].index, inplace=True)
dataset.drop(dataset[dataset['NumOfProducts'] >= 4].index, inplace=True)

print("Shape of Data after removing outliers:", dataset.shape)
```

💬 **نتیجه:**
تعداد رکوردهای پرت کاهش یافته و داده‌ها برای مدل‌سازی واقعی‌تر شده‌اند.

---

### 🧠 ۲ — مهندسی ویژگی‌ها (Feature Engineering)

```python
dataset['BalanceSalary'] = dataset.Balance / dataset.EstimatedSalary
dataset['TenureAge'] = dataset.Tenure / (dataset.Age)
dataset['ScoreAge'] = dataset.CreditScore / (dataset.Age)

dataset['tenure_age'] = dataset.Tenure / (dataset.Age - 17)
dataset['tenure_salary'] = dataset.Tenure / (dataset.EstimatedSalary)
dataset['score_age'] = dataset.CreditScore / (dataset.Age - 17)
dataset['score_salary'] = dataset.CreditScore / (dataset.EstimatedSalary)

dataset["newAge"] = dataset["Age"] - dataset["Tenure"]
dataset["newCreditScore"] = pd.qcut(dataset['CreditScore'], 10, labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
dataset["AgeScore"] = pd.qcut(dataset['Age'], 8, labels = [1, 2, 3, 4, 5, 6, 7, 8])
dataset["BalanceScore"] = pd.qcut(dataset['Balance'].rank(method = "first"), 10, labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
dataset["SalaryScore"] = pd.qcut(dataset['EstimatedSalary'], 10, labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
dataset["newEstimatedSalary"] = dataset["EstimatedSalary"] / 12

dataset['score_balance'] = dataset.Balance / (dataset.CreditScore)
dataset['age_balance'] = dataset.Balance / (dataset.Age)
dataset['balance_salary'] = dataset.Balance / (dataset.EstimatedSalary)
dataset['age_hascrcard'] = dataset.HasCrCard / (dataset.Age)
```

💬 **تحلیل:**
ایجاد نسبت‌هایی مثل «موجودی به حقوق» و «اعتبار به سن» به مدل کمک می‌کند تا روابط پنهان میان شاخص‌های مالی را بهتر بفهمد.

---

### ⚙️ ۳ — تعریف توابع مهندسی پیشرفته

```python
def product_utilization_rate_by_year(row):
    number_of_products = row.NumOfProducts
    tenure = row.Tenure #نرخ استفاده از محصول بر حسب سال‌ها 

    if number_of_products == 0:
        return 0
    if tenure == 0:
        return number_of_products

    rate = number_of_products / tenure
    return rate

def product_utilization_rate_by_salary(row):
    number_of_products = row.NumOfProducts
    estimated_salary = row.EstimatedSalary

    if number_of_products == 0:
        return 0

    rate = number_of_products / estimated_salary
    return rate

def countries_monthly_average_salaries(row):
    fr = 3696
    de = 4740
    sp = 2257
    salary = row.EstimatedSalary / 12
    country = row.Geography

    if country == 'Germany':
        return salary / de
    elif country == "France":
        return salary / fr
    elif country == "Spain":
        return salary / sp
```

---

### 🧮 ۴ — اعمال توابع روی کل داده

```python
def feature_engineering(df):
    df_fe = df.copy()
    df_fe['product_utilization_rate_by_year'] = df_fe.apply(product_utilization_rate_by_year, axis=1)
    df_fe['product_utilization_rate_by_salary'] = df_fe.apply(product_utilization_rate_by_salary, axis=1)
    df_fe['countries_monthly_average_salaries'] = df_fe.apply(countries_monthly_average_salaries, axis=1)
    return df_fe

df_fe = feature_engineering(dataset)
```

---

### 🔠 ۵ — کدگذاری متغیرهای متنی

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df_fe["Geography_le"] = encoder.fit_transform(df_fe["Geography"])
df_fe["Gender_le"] = encoder.fit_transform(df_fe["Gender"])

# One-Hot Encoding
Geography = pd.get_dummies(dataset.Geography, drop_first=True)
Gender = pd.get_dummies(dataset.Gender)

df_fe = pd.concat([df_fe.drop(['Geography', 'Gender'], axis=1),
                   Geography, Gender], axis=1)
```

💬 **تحلیل:**
اکنون تمام متغیرها عددی هستند و برای ورود به مدل‌های یادگیری ترکیبی آماده‌اند.

---

📘 **جمع‌بندی نهایی فصل ۸:**

* داده‌ها پاک‌سازی و نرمال شدند.
* ویژگی‌های جدید و ترکیبی ساخته شد.
* آماده‌سازی برای مرحله‌ی بعدی (نرمال‌سازی و مدل‌سازی Ensemble).


