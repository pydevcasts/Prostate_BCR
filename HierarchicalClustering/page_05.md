**کتاب خوشه‌بندی سلسله‌مراتبی (Hierarchical Clustering)**  
**نویسنده: سیامک عباس‌نژاد**

---

📘 **فصل ۵: تحلیل و ارزیابی نتایج خوشه‌بندی**

تا اینجا خوشه‌بندی سلسله‌مراتبی را روی دیتاست **Iris** اجرا کردیم و سه خوشه به دست آوردیم. حالا می‌خواهیم نتایج را از دو دیدگاه بررسی کنیم:

### 🔹 مقایسه خوشه‌ها با برچسب‌های واقعی دیتاست
```python
from sklearn.metrics import confusion_matrix, adjusted_rand_score

# Confusion matrix between real labels and clusters
cm = confusion_matrix(y, df['cluster'])
print("Confusion Matrix:\n", cm)

# Adjusted Rand Index (ARI)
ari = adjusted_rand_score(y, df['cluster'])
print("Adjusted Rand Index:", ari)
```
📌 **نتیجه**: 
- **ماتریس آشفتگی (Confusion Matrix)** به ما نشان می‌دهد هر کلاس واقعی در چه خوشه‌ای قرار گرفته است.
- **شاخص Adjusted Rand Index (ARI)** بین ۰ و ۱ است. هرچه به ۱ نزدیک‌تر باشد، خوشه‌بندی با برچسب‌های واقعی هماهنگ‌تر است.

### 🔹 ارزیابی کیفیت خوشه‌ها با Silhouette Score
```python
from sklearn.metrics import silhouette_score

# Calculate Silhouette Score
sil_score = silhouette_score(X_scaled, df['cluster'])
print("Silhouette Score:", sil_score)
```
📌 **نتیجه**: 
- **مقدار نزدیک به ۱** → خوشه‌بندی عالی.
- **مقدار نزدیک به ۰** → خوشه‌ها روی هم افتاده‌اند.
- **مقدار منفی** → داده‌ها در خوشه‌های اشتباه قرار گرفته‌اند.

### 🔹 تجسم خوشه‌ها روی دو ویژگی مهم
```python
plt.figure(figsize=(8,6))
sns.scatterplot(x=df["petal length (cm)"], y=df["petal width (cm)"], hue=df["cluster"], palette="Set2", s=80)
plt.title("Clusters Visualization on Petal Features")
plt.show()
```
📌 **نتیجه**: مشاهده می‌کنیم که خوشه‌ها به خوبی جدا شده‌اند و گونه **Setosa** کاملاً مستقل است، اما در بین **Versicolor** و **Virginica** مقداری همپوشانی وجود دارد.

### 🔹 جمع‌بندی
خوشه‌بندی سلسله‌مراتبی روی دیتاست **Iris** توانست سه خوشه مناسب تشکیل دهد.  
مقدار **ARI** نشان داد که این خوشه‌ها با برچسب‌های واقعی تا حد زیادی هم‌پوشانی دارند.  
شاخص **Silhouette** نیز کیفیت خوب خوشه‌بندی را تایید کرد.  
گونه‌ی **Setosa** همیشه به‌راحتی از بقیه جدا می‌شود، در حالی که دو گونه دیگر مقداری همپوشانی دارند.

✅ در این فصل یاد گرفتیم که چطور خوشه‌بندی را به‌صورت کمی و کیفی ارزیابی کنیم.
