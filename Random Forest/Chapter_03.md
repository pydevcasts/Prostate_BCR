## 📖 صفحه ۳: تحلیل آماری اولیه داده‌ها + Boxplot

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 خلاصه آماری داده‌ها

اولین قدم در تحلیل داده‌ها، بررسی شاخص‌های آماری مثل **میانگین، انحراف معیار، حداقل، حداکثر** و سایر مقادیر توصیفی است.

```python
# Summary statistics of dataset
print(df.describe())
```

📊 خروجی نشان می‌دهد که:

* بعضی ویژگی‌ها مثل **mean area** و **mean perimeter** مقدارهای خیلی بزرگ‌تری نسبت به ویژگی‌های دیگر دارند.
* برخی ویژگی‌ها دارای دامنه تغییرات زیادی هستند که می‌تواند روی الگوریتم‌ها تأثیر بگذارد.

این نتایج اهمیت **استانداردسازی (Standardization)** را در مراحل بعدی نشان می‌دهد.

---

### 🔹 نمایش Boxplot برای توزیع ویژگی‌ها

Boxplot یکی از ابزارهای مهم برای تحلیل داده‌هاست چون:

* می‌تواند **پراکندگی داده‌ها** را نشان دهد.
* نقاط پرت (Outliers) را به وضوح مشخص می‌کند.

کد زیر یک نمونه Boxplot از چند ویژگی مهم را رسم می‌کند:

```python
# Boxplot for selected features
plt.figure(figsize=(12,6))
sns.boxplot(data=df[['mean radius', 'mean texture', 'mean area', 'mean smoothness']])
plt.title("Boxplot of Selected Features")
plt.show()
```

📊 نتایج نشان می‌دهد:

* ویژگی **mean area** و **mean radius** دامنه تغییرات بسیار بالاتری دارند.
* برخی ویژگی‌ها مثل **mean smoothness** تغییرات محدودتری دارند.
* نقاط پرت در بعضی ویژگی‌ها وجود دارند که باید در نظر گرفته شوند.

---

### 🔹 مقایسه کلاس‌ها با Boxplot

می‌توانیم بررسی کنیم که توزیع یک ویژگی برای دو کلاس (Benign و Malignant) چه تفاوتی دارد:

```python
plt.figure(figsize=(8,6))
sns.boxplot(x='target', y='mean radius', data=df, palette='Set2')
plt.title("Boxplot of Mean Radius by Class")
plt.xticks([0,1], ['Benign', 'Malignant'])
plt.show()
```

📊 نتیجه:

* مقدار **mean radius** در بیماران Malignant معمولاً بیشتر از Benign است.
* این نشان می‌دهد که **mean radius یکی از ویژگی‌های مهم برای تفکیک کلاس‌هاست**.

---

### 🔹 نتیجه‌گیری

Boxplot نشان داد که:

* برخی ویژگی‌ها مثل **mean area** و **mean radius** نقش مهمی در تمایز کلاس‌ها دارند.
* داده‌ها دارای نقاط پرت هستند که باید مراقب تأثیر آن‌ها بر مدل باشیم.
* تفاوت آماری واضحی بین کلاس‌های Malignant و Benign برای برخی ویژگی‌ها وجود دارد.
