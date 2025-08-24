## 📖 صفحه ۳: معرفی دیتاست Iris و تحلیل داده‌ها

### 🌸 دیتاست Iris چیست؟

دیتاست Iris یکی از مشهورترین دیتاست‌ها در دنیای یادگیری ماشین است که توسط **Ronald Fisher** معرفی شد. این مجموعه داده شامل **۱۵۰ نمونه** گل زنبق در سه گونه است:

* Setosa
* Versicolor
* Virginica

هر گل ۴ ویژگی دارد:

* طول کاسبرگ (Sepal Length)
* عرض کاسبرگ (Sepal Width)
* طول گلبرگ (Petal Length)
* عرض گلبرگ (Petal Width)

هدف ما: **تشخیص گونه گل بر اساس این ویژگی‌ها با استفاده از درخت تصمیم.**

---

### 🔹 گام ۱: بارگذاری و نمایش اولیه داده‌ها

در این مرحله داده‌ها را بارگذاری و ساختار کلی آن‌ها را می‌بینیم.

```python
# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Map numeric target to actual names
df['species'] = df['species'].map({0:'Setosa', 1:'Versicolor', 2:'Virginica'})

# Display first 5 rows
print(df.head())
```

🔍 با این کد یک DataFrame ساخته می‌شود که شامل ۵ سطر اول است. ستون `species` نشان می‌دهد هر گل به کدام گونه تعلق دارد.

---

### 🔹 گام ۲: بررسی آماری اولیه

یکی از بهترین کارها در داده‌کاوی بررسی آماری داده‌ها است.

```python
# Show statistical summary
print(df.describe())
```

خروجی نشان می‌دهد که مثلاً **گلبرگ Setosa** خیلی کوچک‌تر از دو گونه دیگر است. این همان چیزی است که بعدها در درخت تصمیم به‌عنوان یک معیار مهم استفاده خواهد شد. 🌱

---

### 🔹 گام ۳: Heatmap (نقشه حرارتی همبستگی)

برای بررسی ارتباط بین ویژگی‌ها، از **ماتریس همبستگی (Correlation Matrix)** استفاده می‌کنیم.

```python
# Compute correlation matrix
corr = df.drop('species', axis=1).corr()

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Heatmap of Feature Correlations", fontsize=14)
plt.show()
```

📊 **تفسیر Heatmap:**

* همبستگی بین طول گلبرگ و عرض گلبرگ بسیار بالا است (رنگ نزدیک به قرمز).
* عرض کاسبرگ کمتر همبسته است و ممکن است ویژگی کم‌اهمیت‌تری باشد.
* این اطلاعات کمک می‌کند بفهمیم کدام ویژگی‌ها برای تفکیک گونه‌ها مهم‌تر هستند.

---

### 🔹 گام ۴: Boxplot (جعبه‌نمودار توزیع ویژگی‌ها)

Boxplot کمک می‌کند توزیع داده‌های هر ویژگی برای سه گونه مختلف بررسی شود.

```python
# Plot Boxplots for all features
plt.figure(figsize=(12,8))

for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(2,2,i)
    sns.boxplot(x="species", y=col, data=df, palette="Set2")
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()
```

📊 **تفسیر Boxplot:**

* طول گلبرگ (Petal Length) به وضوح گونه Setosa را از دو گونه دیگر جدا می‌کند.
* عرض کاسبرگ (Sepal Width) بین گونه‌ها همپوشانی بیشتری دارد، پس کمتر درخت تصمیم را کمک خواهد کرد.
* به طور کلی، ویژگی‌های گلبرگ (Petal) قوی‌تر از کاسبرگ (Sepal) هستند. 🌸

---

### 🔹 گام ۵: شناسایی ویژگی‌های مهم

در همین مرحله حتی بدون ساخت مدل، از روی نمودارها می‌توان نتیجه گرفت:

1. **ویژگی‌های گلبرگ (Petal Length, Petal Width)** بیشترین قدرت جداسازی گونه‌ها را دارند.
2. **ویژگی‌های کاسبرگ (Sepal Length, Sepal Width)** کمتر تأثیرگذار هستند.

این یافته‌ها بعدها هنگام آموزش مدل Decision Tree تأیید خواهند شد.

---

### 🎯 نتیجه این صفحه

* دیتاست Iris شامل ۱۵۰ گل با ۴ ویژگی است.
* بررسی آماری نشان داد داده‌ها ساختار مناسبی دارند.
* Heatmap نشان داد ویژگی‌های گلبرگ ارتباط قوی‌تری دارند.
* Boxplot ثابت کرد ویژگی‌های گلبرگ گونه‌ها را بهتر جدا می‌کنند.