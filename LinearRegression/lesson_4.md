
# 📘 فصل ۳: پروژه واقعی – پیش‌بینی قیمت خانه با رگرسیون چندمتغیره

### ✅ **صفحه ۴ – آماده‌سازی داده‌ها، تحلیل اولیه و انتخاب ویژگی‌ها**

---

## 🎯 هدف این صفحه:

در این بخش، قصد داریم یک پروژه واقعی پیاده‌سازی کنیم.
با استفاده از **دیتاست استاندارد California Housing**، می‌خواهیم قیمت خانه را بر اساس چند ویژگی مختلف **مدل‌سازی و پیش‌بینی** کنیم.

در این صفحه:

* داده‌ها را بارگذاری و پاک‌سازی می‌کنیم
* تحلیل همبستگی بین ویژگی‌ها انجام می‌دهیم
* متغیرهای مفید برای مدل را انتخاب می‌کنیم

---

## 📂 ۱. بارگذاری دیتاست California Housing

Scikit-learn یک دیتاست آماده دارد به نام **California Housing** که برای آموزش رگرسیون ایده‌آل است.

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

# Display the first few rows
print(df.head())
```

---

### 🧾 داده‌ها شامل ویژگی‌های زیر هستند:

| Feature      | Description                                      |
| ------------ | ------------------------------------------------ |
| `MedInc`     | Median income in block                           |
| `HouseAge`   | Average house age                                |
| `AveRooms`   | Average number of rooms                          |
| `AveBedrms`  | Average number of bedrooms                       |
| `Population` | Block population                                 |
| `AveOccup`   | Average occupancy per household                  |
| `Latitude`   | Geo latitude                                     |
| `Longitude`  | Geo longitude                                    |
| `Target`     | Median house value (in \$100,000s) ✅ **هدف مدل** |

---

## 🔍 ۲. بررسی اولیه داده‌ها

```python
# Check basic statistics
print(df.describe())
```

**نکاتی که بررسی می‌کنیم:**

* آیا داده‌های منفی یا غیرمنطقی وجود دارد؟
* توزیع مقادیر در ستون هدف (`Target`) چگونه است؟
* میانگین و انحراف معیار ویژگی‌ها چقدر است؟

---

## ❗ ۳. آیا داده گمشده داریم؟

قبل از هر کاری باید مطمئن شویم دیتاست ما پاک و قابل استفاده است.

```python
# Check for missing values
print(df.isnull().sum())
```

📌 در دیتاست California Housing معمولاً هیچ داده گمشده‌ای وجود ندارد — اما این بررسی همیشه باید انجام شود.

---

## 📊 ۴. تحلیل همبستگی ویژگی‌ها (Correlation)

برای اینکه بفهمیم **کدام ویژگی‌ها بیشترین تأثیر را روی قیمت دارند**، نمودار همبستگی می‌سازیم.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
```

---

### 🔍 نکات قابل مشاهده در Heatmap:

* `MedInc` (درآمد میانگین) بیشترین همبستگی مثبت با قیمت دارد
* `AveRooms` نیز همبستگی دارد، اما باید دقت کرد که ممکن است با `AveBedrms` هم‌خطی داشته باشد
* `Latitude` و `Longitude` معمولاً تأثیر غیرخطی دارند (موقعیت جغرافیایی)

---

## 🧩 ۵. انتخاب ویژگی‌های کلیدی

براساس همبستگی و درک مفهومی، ویژگی‌های زیر برای مدل انتخاب می‌شوند:

```python
selected_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveOccup']
```

### 📌 چرا این انتخاب؟

* `MedInc`: چون قوی‌ترین پیش‌بینی‌کننده قیمت است
* `HouseAge`: سن ساختمان معمولاً با قیمت رابطه دارد (مشتریان ترجیح‌های خاص دارند)
* `AveRooms`: نشان‌دهنده وسعت خانه است
* `AveOccup`: تراکم خانوار ممکن است شاخص کیفیت زندگی باشد

---

## ✅ ۶. آماده‌سازی داده برای آموزش مدل

```python
from sklearn.model_selection import train_test_split

# Select features and target
X = df[selected_features]
y = df['Target']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
📌 **۲۰٪ داده‌ها را برای تست کنار می‌گذاریم** تا مدل را روی داده‌های نادیده‌گرفته‌شده ارزیابی کنیم.
