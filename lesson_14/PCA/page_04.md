
---

## 📖 صفحه ۴: تحلیل داده‌ها (EDA قوی‌تر)

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 اهمیت تحلیل داده

قبل از اجرای PCA، باید بدانیم که داده‌ها چه ساختاری دارند. اگر بدون بررسی سراغ کاهش بعد برویم، ممکن است الگوهای مهم یا مشکلاتی مثل **outlier** و **ویژگی‌های کم‌اهمیت** را نادیده بگیریم. بنابراین، تحلیل داده‌ها (EDA: Exploratory Data Analysis) گام حیاتی در هر پروژه Data Science است.

---

### 🔹 بررسی توزیع ویژگی‌ها (Histogram)

```python
import matplotlib.pyplot as plt

data.drop('target', axis=1).hist(bins=20, figsize=(15,12), color='skyblue', edgecolor='black')
plt.suptitle("Distribution of Wine Features", fontsize=16)
plt.show()
```

📌 تحلیل:

* برخی ویژگی‌ها مثل **Alcohol** و **Proline** توزیع نسبتاً نرمال دارند.
* ویژگی‌هایی مثل **Malic acid** یا **Color intensity** دارای توزیع غیرمتقارن (Skewed) هستند.
* این نکته در اجرای PCA مهم است، چون PCA به واریانس ویژگی‌ها حساس است.

---

### 🔹 بررسی Outlierها با Boxplot

```python
plt.figure(figsize=(15,10))
data.drop('target', axis=1).boxplot(rot=90)
plt.title("Boxplot of Wine Features")
plt.show()
```

📌 تحلیل:

* ویژگی **Malic acid** و **Color intensity** دارای outlierهای قابل توجه هستند.
* وجود outlierها می‌تواند روی PCA اثر بگذارد چون مقادیر بزرگ و غیرمعمول، واریانس را تغییر می‌دهند.

---

### 🔹 بررسی همبستگی ویژگی‌ها با Heatmap

```python
import seaborn as sns

plt.figure(figsize=(12,10))
sns.heatmap(data.drop('target', axis=1).corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap of Wine Features")
plt.show()
```

📌 تحلیل:

* ویژگی **Flavanoids** همبستگی مثبت بالایی با **Total phenols** دارد.
* ویژگی **Color intensity** همبستگی منفی با **Hue** دارد.
* چنین همبستگی‌هایی نشان می‌دهد که برخی ویژگی‌ها اطلاعات تکراری دارند و PCA می‌تواند آن‌ها را ترکیب کند.

---

### 🔹 مقایسه میانگین ویژگی‌ها در کلاس‌های مختلف (Bar Plot)

```python
plt.figure(figsize=(12,6))
class_means = data.groupby('target').mean()
class_means.T.plot(kind='bar', figsize=(15,8))
plt.title("Mean of Features per Class")
plt.xlabel("Features")
plt.ylabel("Mean Value")
plt.legend(title="Wine Class")
plt.show()
```

📌 تحلیل:

* ویژگی **Alcohol** بین کلاس‌ها تفاوت چشمگیری دارد.
* ویژگی **Proline** نیز نقش مهمی در تمایز کلاس‌ها ایفا می‌کند.
* برخی ویژگی‌ها مثل **Magnesium** در کلاس‌ها تفاوت زیادی ندارند و احتمالاً نقش کمی در جداسازی دارند.

---

### 🔹 نتیجه تحلیل داده‌ها

1. برخی ویژگی‌ها دارای توزیع نامتقارن و Outlier هستند.
2. همبستگی بالا بین برخی ویژگی‌ها وجود دارد (تکراری بودن اطلاعات).
3. چند ویژگی مانند Alcohol و Proline نقش مهمی در تمایز کلاس‌ها دارند.

📌 این نتایج نشان می‌دهند که **PCA می‌تواند به ما کمک کند ویژگی‌های مهم‌تر را نگه داریم و ویژگی‌های همبسته یا کم‌اهمیت را کنار بگذاریم.**

