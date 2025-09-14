## 📖 صفحه ۱۲: نمایش مقایسه نتایج مدل‌ها با نمودارها

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 چرا نمایش تصویری مقایسه مدل‌ها مهم است؟

* اعداد دقت (Accuracy) فقط یک نگاه کلی می‌دهند.
* نمودارها به ما کمک می‌کنند تا **پایداری** (Stability) و **مقایسه مستقیم بین مدل‌ها** را بهتر ببینیم.
* با Boxplot می‌توانیم نوسانات و Outlierها را مشاهده کنیم.

---

### 🔹 Barplot از میانگین دقت‌ها

```python
# Prepare mean accuracy for each model
mean_scores = {name: scores.mean() for name, scores in cv_results.items()}

plt.figure(figsize=(8,6))
sns.barplot(x=list(mean_scores.keys()), y=list(mean_scores.values()), palette="Set2")
plt.title("Mean Accuracy of Models (Cross Validation)")
plt.ylabel("Mean Accuracy")
plt.ylim(0.9, 1.0)
plt.show()
```

📊 تفسیر:

* Random Forest بالاترین ستون را دارد (حدود ۹۶٪).
* KNN کوتاه‌تر است (حدود ۹۴٪).
* Logistic Regression و SVM نزدیک به هم هستند.

---

### 🔹 Boxplot از نتایج Cross Validation

```python
# Convert results to DataFrame for visualization
cv_df = pd.DataFrame(cv_results)

plt.figure(figsize=(8,6))
sns.boxplot(data=cv_df, palette="Set3")
plt.title("Accuracy Distribution of Models (Cross Validation)")
plt.ylabel("Accuracy")
plt.ylim(0.9, 1.0)
plt.show()
```

📊 تفسیر:

* جعبه Random Forest کوچک‌تر است → نوسانات کمتر و پایداری بیشتر.
* KNN جعبه بلندتری دارد → نوسان بیشتر و عدم پایداری در Foldهای مختلف.
* Logistic Regression و SVM پایداری نسبتاً خوبی دارند، ولی باز هم کمی پایین‌تر از Random Forest.

---

### 🔹 نتیجه‌گیری

* **Random Forest بهترین مدل** در این دیتاست است (هم دقت بالا و هم پایداری عالی).
* **SVM و Logistic Regression** انتخاب‌های خوب و ساده‌تری هستند.
* **KNN** ضعیف‌تر عمل کرده و نوسان بیشتری دارد.

---

### 🔹 مقدمه برای ادامه

در صفحه بعد می‌توانیم تحلیل پیشرفته‌تری داشته باشیم:

* مقایسه **Precision، Recall، F1-Score** بین مدل‌ها.
* نمایش این مقایسه با نمودار Heatmap یا Barplot.

