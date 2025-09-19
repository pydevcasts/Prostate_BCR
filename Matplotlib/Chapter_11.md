## 📖 فصل ۱۱: پروژه نهایی – تحلیل داده واقعی

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 معرفی پروژه

در این پروژه یک دیتاست واقعی بررسی می‌کنیم. دیتاست انتخابی ما: **Iris Dataset**
(یکی از معروف‌ترین دیتاست‌های یادگیری ماشین برای طبقه‌بندی گل‌ها).

هدف:

1. تحلیل داده‌ها
2. استفاده از نمودارهای آماری و بصری
3. ترکیب **Matplotlib، Pandas و Seaborn**

---

### 🔹 ۱. بارگذاری و بررسی اولیه داده‌ها

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# بارگذاری دیتاست
iris = sns.load_dataset("iris")

# مشاهده چند ردیف اول
print(iris.head())

# آمار توصیفی
print(iris.describe())
```

📌 دیتاست شامل ۴ ویژگی (Sepal Length, Sepal Width, Petal Length, Petal Width) و یک ستون برچسب (Species) است.

---

### 🔹 ۲. هیستوگرام ویژگی‌ها

```python
iris.hist(figsize=(10, 8), color="skyblue", edgecolor="black")
plt.suptitle("Histogram of Iris Features")
plt.show()
```

📌 این نمودار توزیع هر ویژگی را نشان می‌دهد.

---

### 🔹 ۳. Boxplot برای مقایسه بین گونه‌ها

```python
sns.boxplot(x="species", y="sepal_length", data=iris, palette="Set2")
plt.title("Sepal Length by Species")
plt.show()
```

📌 به وضوح تفاوت بین گونه‌ها در طول کاسبرگ دیده می‌شود.

---

### 🔹 ۴. Pairplot برای بررسی ارتباط ویژگی‌ها

```python
sns.pairplot(iris, hue="species", diag_kind="kde", palette="husl")
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()
```

📌 این نمودار پراکندگی ویژگی‌ها را نمایش می‌دهد و نشان می‌دهد گونه‌ها تا حد زیادی قابل تفکیک هستند.

---

### 🔹 ۵. Heatmap ماتریس همبستگی

```python
corr = iris.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
```

📌 بیشترین همبستگی بین طول و عرض گلبرگ‌ها دیده می‌شود.

---

### 🔹 ۶. Violin Plot برای نمایش توزیع دقیق‌تر

```python
sns.violinplot(x="species", y="petal_length", data=iris, palette="muted")
plt.title("Petal Length Distribution by Species")
plt.show()
```

📌 شکل توزیع داده‌ها (نه فقط میانه و چارک‌ها) مشخص می‌شود.

---

### 🔹 نتیجه‌گیری پروژه

* **Histogram** → توزیع داده‌ها.
* **Boxplot و Violinplot** → مقایسه آماری بین گونه‌ها.
* **Pairplot** → نمایش ارتباط ویژگی‌ها.
* **Heatmap** → بررسی همبستگی ویژگی‌ها.

این ترکیب ابزارها نشان می‌دهد چگونه می‌توان داده‌های واقعی را با **Matplotlib + Seaborn + Pandas** تحلیل کرد.
