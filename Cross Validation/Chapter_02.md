
## 📖 صفحه ۲: معرفی Wine Dataset و آماده‌سازی داده‌ها

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 معرفی Wine Dataset

یکی از دیتاست‌های کلاسیک و پرکاربرد در یادگیری ماشین، دیتاست **Wine** است که برای تشخیص نوع شراب بر اساس ویژگی‌های شیمیایی آن استفاده می‌شود.

این دیتاست شامل **۱۷۸ نمونه** و **۱۳ ویژگی عددی** است. هر نمونه به یکی از **۳ کلاس مختلف** تعلق دارد (یعنی نوع شراب).

---

### 🔹 ویژگی‌های اصلی دیتاست

برخی از ستون‌های مهم:

* Alcohol (میزان الکل)
* Malic Acid (اسید مالیک)
* Ash (خاکستر)
* Magnesium (منیزیم)
* Flavanoids (فلاونوئیدها)
* Proline (پروتئین‌ها)

📌 هدف ما این است که با استفاده از این ویژگی‌ها، **مدل یادگیری ماشین بسازیم که نوع شراب را پیش‌بینی کند**.

---

### 🔹 بارگذاری و بررسی اولیه داده‌ها

```python
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# Load dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name="target")

# Show first 5 rows
print(X.head())
print(y.head())
```

---

### 🔹 بررسی آماری اولیه

```python
# Statistical summary
print(X.describe())
```

📊 این خروجی به ما میانگین، میانه، انحراف معیار و بازه تغییرات هر ویژگی را نشان می‌دهد.

---

### 🔹 تحلیل بصری اولیه

برای درک بهتر داده‌ها از نمودارها استفاده می‌کنیم:

#### ۱. نمودار Boxplot برای توزیع ویژگی‌ها

```python
# Boxplot for all features
plt.figure(figsize=(15,8))
sns.boxplot(data=X)
plt.xticks(rotation=90)
plt.title("Boxplot of Wine Features")
plt.show()
```

📌 این نمودار به ما نشان می‌دهد که داده‌ها چه مقدار پراکندگی دارند و آیا نقاط پرت (Outliers) وجود دارد یا خیر.

---

#### ۲. Heatmap برای همبستگی ویژگی‌ها

```python
# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(X.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap of Wine Features")
plt.show()
```

📌 Heatmap مشخص می‌کند کدام ویژگی‌ها با هم ارتباط قوی دارند. مثلاً در Wine Dataset، ویژگی **Flavanoids** معمولاً همبستگی بالایی با **OD280/OD315** دارد.

---

### 🔹 جمع‌بندی صفحه ۲

در این مرحله:

* داده‌ها را بارگذاری کردیم.
* یک نگاه آماری اولیه داشتیم.
* با استفاده از Boxplot و Heatmap، پراکندگی و ارتباط ویژگی‌ها را بررسی کردیم.

این تحلیل اولیه کمک می‌کند قبل از ساخت مدل، **درک بهتری از داده‌ها** داشته باشیم.
