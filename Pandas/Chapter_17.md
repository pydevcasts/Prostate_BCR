# 📖 فصل ۱۷: پروژه نهایی – تحلیل یک دیتاست واقعی با Pandas

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. مقدمه

در این فصل قصد داریم همه‌ی مفاهیمی که در فصل‌های قبل یاد گرفتیم (مثل **خواندن داده، پاک‌سازی، انتخاب ستون‌ها، گروه‌بندی، مصورسازی، و بهینه‌سازی حافظه**) را در یک پروژه عملی روی یک دیتاست واقعی به کار ببریم.
برای این کار از دیتاست معروف **Titanic** استفاده می‌کنیم.

این دیتاست شامل اطلاعات مسافران کشتی تایتانیک است:

* نام
* جنسیت
* سن
* کلاس بلیت
* وضعیت زنده ماندن

---

### 🔹 ۲. بارگذاری داده‌ها

```python
import pandas as pd

# Load Titanic dataset
df = pd.read_csv("titanic.csv")

# Display first rows
print(df.head())
```

---

### 🔹 ۳. بررسی اولیه داده‌ها

```python
# Summary of dataset
print(df.info())

# Check missing values
print(df.isnull().sum())

# convert str Sex to boolean 
df["Sex"]= df["Sex"].str.lower().map({"man":1,"female":0})

```

📌 می‌بینیم ستون‌هایی مثل **Age** و **Cabin** داده‌های ناقص دارند.

---

### 🔹 ۴. پاک‌سازی و پیش‌پردازش

```python
# Fill missing ages with median
df["Age"].fillna(df["Age"].median(), inplace=True)

# Drop 'Cabin' column because too many missing values
df.drop(columns=["Cabin"], inplace=True)

# Fill missing Embarked values with mode
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
```

---

### 🔹 ۵. تحلیل داده‌ها

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Survival count
sns.countplot(data=df, x="Survived")
plt.title("Survival Distribution")
plt.show()

# Age distribution by survival
sns.boxplot(data=df, x="Survived", y="Age")
plt.title("Age vs Survival")
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```

📊 نتایج:

* زنان بیشتر از مردان شانس زنده ماندن داشتند
* مسافران کلاس ۱ بیشتر زنده ماندند
* سن در زنده ماندن نقش نسبی داشت

---

### 🔹 ۶. گروه‌بندی و آمار

```python
# Survival rate by class
print(df.groupby("Pclass")["Survived"].mean())

# Survival rate by gender
print(df.groupby("Sex")["Survived"].mean())
```

---

### 🔹 ۷. بهینه‌سازی حافظه

```python
# Convert object to category
df["Sex"] = df["Sex"].astype("category")
df["Embarked"] = df["Embarked"].astype("category")

# Convert integer columns to smaller types
df["Pclass"] = pd.to_numeric(df["Pclass"], downcast="integer")
df["Survived"] = pd.to_numeric(df["Survived"], downcast="integer")
```

---

### 🔹 ۸. نتیجه‌گیری پروژه

با استفاده از Pandas توانستیم:

* داده‌ها را بخوانیم و پاک‌سازی کنیم
* داده‌های گمشده را مدیریت کنیم
* آمار توصیفی و گروه‌بندی انجام دهیم
* الگوهای مهم را کشف کنیم (مثلاً تأثیر جنسیت و کلاس بلیت در بقا)
* از مصورسازی‌ها برای درک بهتر داده استفاده کنیم
* مصرف حافظه را بهینه کنیم

---
