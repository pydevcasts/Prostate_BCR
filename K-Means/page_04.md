# 📖 فصل چهارم: پیاده‌سازی K-Means روی دیتاست Iris

---

## 🔹 مرحله اول: بارگذاری دیتاست Iris

ابتدا دیتاست رو از **scikit-learn** بارگذاری می‌کنیم و یه نگاه اولیه بهش می‌ندازیم.

```python
# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Show first rows
print(df.head())

# Show dataset info
print(df.info())
```

---

## 🔹 مرحله دوم: تحلیل داده‌ها (EDA)

اینجا با استفاده از نمودارها توزیع ویژگی‌ها و روابط بین آن‌ها رو بررسی می‌کنیم.

### ۱. توزیع هر ویژگی (Histogram)

```python
# Plot histogram for each feature
df.iloc[:, :-1].hist(figsize=(10, 8), bins=15, color='skyblue')
plt.suptitle("Distribution of Iris Features", fontsize=16)
plt.show()
```

🔎 توضیح:

* Setosa معمولاً طول کاسبرگ (sepal length) کوتاه‌تری نسبت به دو گونه دیگر داره.
* Petal length (طول گلبرگ) خیلی خوب گونه‌ها رو از هم جدا می‌کنه.

---

### ۲. Boxplot برای مقایسه ویژگی‌ها بین گونه‌ها

```python
# Plot boxplots for all features
plt.figure(figsize=(12, 8))
for i, col in enumerate(df.columns[:-1]):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x="target", y=col, data=df)
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()
```

🔎 توضیح:

* **Petal length** و **Petal width** تفکیک خیلی واضحی بین Setosa و بقیه ایجاد می‌کنن.
* Sepal width تنوع بیشتری داره و کمتر قابل تفکیکه.

---

### ۳. Heatmap برای همبستگی ویژگی‌ها

```python
# Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.show()
```

🔎 توضیح:

* بالاترین همبستگی بین **petal length** و **petal width** هست (\~0.96).
* Sepal length با Petal length همبستگی متوسطی داره (\~0.87).

---

### ۴. Pairplot برای نمایش خوشه‌های واقعی

```python
# Pairplot with species labels
sns.pairplot(df, hue="target", diag_kind="hist", palette="Set2")
plt.suptitle("Pairplot of Iris Dataset", y=1.02, fontsize=16)
plt.show()
```

🔎 توضیح:

* Setosa (کلاس ۰) به‌وضوح از دو کلاس دیگه جداست.
* Versicolor و Virginica مقداری هم‌پوشانی دارن.

---

✅ تا اینجا مرحله اول و دوم رو با تحلیل قدرتمند انجام دادیم.
از داده‌ها متوجه شدیم که **Petal length** و **Petal width** ویژگی‌های کلیدی برای خوشه‌بندی هستن.

