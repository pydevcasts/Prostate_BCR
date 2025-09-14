## 📖 صفحه ۵: تحلیل دقیق‌تر Logistic Regression

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 اهمیت بررسی پایداری مدل

وقتی یک مدل مثل Logistic Regression در Cross Validation دقت بالا دارد، باید بررسی کنیم که **چقدر نتایج پایدار هستند**. اگر دقت در Foldها خیلی متغیر باشد، یعنی مدل روی داده‌های مختلف رفتار متفاوتی دارد و پایدار نیست.

---

### 🔹 آماره‌های توصیفی برای نتایج Cross Validation

```python
print("Mean Accuracy:", np.mean(scores_logreg))
print("Standard Deviation:", np.std(scores_logreg))
print("Min Accuracy:", np.min(scores_logreg))
print("Max Accuracy:", np.max(scores_logreg))
```

📌 تفسیر:

* **میانگین (Mean):** دقت کلی مدل.
* **انحراف معیار (Std):** هرچه کوچک‌تر باشد، مدل پایدارتر است.
* **کمینه و بیشینه:** نشان‌دهنده بهترین و بدترین عملکرد مدل در Foldها.

---

### 🔹 نمودار Boxplot برای دقت‌ها

```python
plt.figure(figsize=(6,5))
sns.boxplot(y=scores_logreg, color="skyblue")
sns.stripplot(y=scores_logreg, color="red", size=8, jitter=True)
plt.title("Boxplot of Logistic Regression Accuracy in Cross Validation")
plt.ylabel("Accuracy")
plt.show()
```

📊 **Boxplot** به ما نشان می‌دهد:

* توزیع دقت‌ها در Cross Validation
* میانه (خط وسط جعبه)
* دامنه تغییرات (Whiskers)
* نقاط پرت (اگر وجود داشته باشند)

---

### 🔹 تفسیر احتمالی نتایج

اگر انحراف معیار بسیار کم باشد (مثلاً زیر ۰.۰۲)، یعنی Logistic Regression برای دیتاست Wine بسیار **پایدار** است.
این موضوع نشان می‌دهد که داده‌ها به خوبی قابل جداسازی هستند و Logistic Regression انتخاب مناسبی برای این دیتاست است.

---

### 🔹 نکته کلیدی برای دانشجویان

* داشتن **دقت بالا** به تنهایی کافی نیست؛ باید **پایداری** مدل هم بررسی شود.
* Cross Validation یکی از ابزارهای اصلی برای رسیدن به این هدف است.
* نمودارهایی مثل Boxplot یا Barplot کمک می‌کنند که نتایج بهتر درک شوند.

---

### 🔹 جمع‌بندی صفحه ۵

* دقت Logistic Regression را با آمار توصیفی (میانگین، انحراف معیار، کمینه و بیشینه) بررسی کردیم.
* با نمودار Boxplot پایداری مدل را به‌صورت بصری تحلیل کردیم.
* نتیجه: Logistic Regression علاوه بر دقت بالا، پایداری بسیار خوبی در دیتاست Wine دارد.

