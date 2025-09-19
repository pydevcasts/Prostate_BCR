# 📘 کتاب: خوشه‌بندی (Clustering)

✍️ نویسنده: سیامک عباس‌نژاد

---

## 📖 فصل ۱: مقدمه‌ای بر خوشه‌بندی

خوشه‌بندی یکی از مهم‌ترین تکنیک‌های یادگیری بدون نظارت است.
در این روش، داده‌ها **بدون داشتن برچسب (Label)** به گروه‌هایی تقسیم می‌شوند که درون هر گروه شباهت زیادی بین داده‌ها وجود دارد و بین گروه‌ها تفاوت‌ها بیشتر است.

### مثال ساده:

فرض کنید مجموعه‌ای از مشتریان یک فروشگاه را داریم. ما نمی‌دانیم کدام مشتری «پُرخرج» است یا «صرفه‌جو». خوشه‌بندی به ما کمک می‌کند مشتریان را بر اساس الگوهای خریدشان به گروه‌های مشابه تقسیم کنیم.

---

## 📐 فرمول پایه‌ای خوشه‌بندی K-Means

در الگوریتم K-Means، ما باید **K مرکز خوشه (Centroid)** تعریف کنیم.
هدف: کمینه کردن مجموع فاصله‌ی داده‌ها از نزدیک‌ترین مرکز خوشه.

$$
J = \sum_{i=1}^{K} \sum_{x \in C_i} ||x - \mu_i||^2
$$

* $K$: تعداد خوشه‌ها
* $C_i$: مجموعه نقاط در خوشه $i$
* $\mu_i$: مرکز خوشه $i$

---

## 📊 دیتاست پیشنهادی برای پروژه

ما از دیتاست **Iris** شروع می‌کنیم چون ۳ کلاس داره (Setosa, Versicolor, Virginica) و می‌تونیم با خوشه‌بندی ببینیم چطور داده‌ها به‌صورت خودکار گروه‌بندی می‌شن.

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target for visualization
df['target'] = iris.target

# Quick view
df.head()
```

---

## 📊 تحلیل داده‌ها با نمودارها

```python
# Pairplot for data distribution
sns.pairplot(df, hue="target", diag_kind="kde")
plt.show()
```

📌 در این نمودار می‌بینیم که کلاس Setosa از دو کلاس دیگر جداست، اما Versicolor و Virginica همپوشانی زیادی دارن. همین موضوع چالش خوشه‌بندی رو نشون می‌ده.

