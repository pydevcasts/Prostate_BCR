**کتاب خوشه‌بندی سلسله‌مراتبی (Hierarchical Clustering)**  
**نویسنده: سیامک عباس‌نژاد**

---

📘 **فصل ۴: پیاده‌سازی خوشه‌بندی سلسله‌مراتبی با پایتون**

در این فصل با استفاده از دیتاست معروف **Iris**، الگوریتم خوشه‌بندی سلسله‌مراتبی را پیاده‌سازی و نتایج آن را تحلیل می‌کنیم.

### 🔹 مرحله ۱: بارگذاری کتابخانه‌ها و داده‌ها
```python
# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df.head()
```

### 🔹 مرحله ۲: تحلیل داده‌ها (EDA)
#### توزیع ویژگی‌ها با Boxplot
```python
# Plot boxplots for each feature
plt.figure(figsize=(12,6))
df.drop("target", axis=1).boxplot()
plt.title("Boxplot of Iris Features")
plt.show()
```
📌 **نتیجه**: نمودار باکس‌پلات به ما نشان می‌دهد که ویژگی‌ها در چه بازه‌ای قرار دارند و آیا داده‌های پرت (outliers) وجود دارند یا نه.

#### بررسی همبستگی ویژگی‌ها با Heatmap
```python
# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.drop("target",axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Features")
plt.show()
```
📌 **نتیجه**: برخی ویژگی‌ها مثل طول و عرض گلبرگ همبستگی بالایی دارند، پس احتمالاً نقش کلیدی در خوشه‌بندی خواهند داشت.

### 🔹 مرحله ۳: استانداردسازی داده‌ها
```python
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
📌 **نتیجه**: چون ویژگی‌ها در مقیاس‌های مختلف هستند (مثلاً میلی‌متر، سانتی‌متر)، باید آن‌ها را استاندارد کنیم تا الگوریتم دچار خطا نشود.

### 🔹 مرحله ۴: اجرای خوشه‌بندی سلسله‌مراتبی
```python
# Perform hierarchical clustering with Ward method
Z = linkage(X_scaled, method="ward")
```

### 🔹 مرحله ۵: رسم دندروگرام
```python
plt.figure(figsize=(12,6))
dendrogram(Z, labels=iris.target, leaf_rotation=90)
plt.title("Hierarchical Clustering Dendrogram (Ward)")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()
```
📌 **نتیجه**: در دندروگرام مشاهده می‌کنیم که داده‌ها چگونه به مرور ادغام می‌شوند. با برش در ارتفاع مناسب می‌توان تعداد خوشه‌ها را تعیین کرد.

### 🔹 مرحله ۶: انتخاب تعداد خوشه‌ها
```python
# Cut the dendrogram at height to form clusters
clusters = fcluster(Z, t=3, criterion='maxclust')

# Add predicted clusters to dataframe
df['cluster'] = clusters
```

### 🔹 مرحله ۷: تجسم نتایج
```python
# Pairplot for cluster visualization
sns.pairplot(df, vars=iris.feature_names, hue="cluster", palette="Set1")
plt.show()
```
📌 **نتیجه**: داده‌ها به سه خوشه اصلی تقسیم شده‌اند که تقریباً با کلاس‌های واقعی (Setosa, Versicolor, Virginica) هم‌پوشانی دارند.

✅ در این فصل توانستیم با استفاده از پایتون و دیتاست **Iris**، الگوریتم خوشه‌بندی سلسله‌مراتبی را پیاده‌سازی کرده و با ابزارهای تصویری مختلف تحلیل کنیم.

