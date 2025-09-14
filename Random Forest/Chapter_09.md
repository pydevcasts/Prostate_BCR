## 📖 صفحه ۹: نمایش نتایج Cross Validation با نمودارها

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 چرا تصویرسازی نتایج؟

اعداد دقت (Accuracy) برای هر Fold اطلاعات خوبی می‌دهند، اما وقتی این نتایج را به شکل نمودار نمایش دهیم:

* درک پایداری مدل ساده‌تر می‌شود.
* می‌توانیم مقایسه بصری بین Foldها داشته باشیم.
* نوسانات عملکرد مدل سریع‌تر شناسایی می‌شوند.

---

### 🔹 Barplot برای دقت در هر Fold

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Barplot of accuracy scores
plt.figure(figsize=(8,6))
sns.barplot(x=list(range(1, len(scores)+1)), y=scores, palette="Blues_d")
plt.title("Random Forest Accuracy per Fold")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0.9, 1.0)
plt.show()
```

📊 تفسیر:

* بیشتر Foldها دقتی بین **۹۴٪ تا ۹۷٪** دارند.
* تفاوت بین Foldها بسیار کم است → پایداری بالا.

---

### 🔹 Boxplot برای توزیع دقت‌ها

```python
# Boxplot of accuracy scores
plt.figure(figsize=(6,5))
sns.boxplot(y=scores, color="lightgreen")
plt.title("Distribution of Accuracy Scores (Random Forest)")
plt.ylabel("Accuracy")
plt.show()
```

📊 تفسیر:

* میانگین دقت نزدیک به ۹۶٪ است.
* محدوده بین چارکی (IQR) بسیار کوچک است → نوسان کم.
* هیچ Outlier قابل توجهی وجود ندارد → مدل در همه Foldها عملکرد مشابهی داشته است.

---

### 🔹 مقایسه Barplot و Boxplot

* **Barplot** نشان داد که همه Foldها عملکرد تقریباً یکسان دارند.
* **Boxplot** این پایداری را تأیید کرد و نشان داد که Random Forest یک مدل پایدار و مقاوم برای این دیتاست است.

---

### 🔹 نتیجه‌گیری

* Random Forest روی دیتاست Breast Cancer دقت بالای **۹۶٪** با پایداری عالی دارد.
* نمایش نتایج با نمودارها به درک بهتر عملکرد مدل کمک کرد.
