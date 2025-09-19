## 📖 صفحه ۴: انتخاب ویژگی‌های مهم + تحلیل بصری ویژگی‌ها

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 چرا انتخاب ویژگی مهم است؟

در پروژه‌های واقعی یادگیری ماشین، تعداد ویژگی‌ها (Features) می‌تواند بسیار زیاد باشد. اما همه‌ی ویژگی‌ها به یک اندازه در تشخیص **Spam** یا **Ham** تأثیرگذار نیستند. انتخاب ویژگی‌های مهم (Feature Selection) باعث می‌شود:

* مدل ساده‌تر و سریع‌تر آموزش ببیند.
* از Overfitting جلوگیری شود.
* دقت مدل بهبود پیدا کند.

---

### 🔹 محاسبه اهمیت ویژگی‌ها با روش آماری ساده

یکی از روش‌های ساده برای بررسی اهمیت ویژگی‌ها، محاسبه **میانگین مقدار ویژگی در ایمیل‌های Spam و Ham** و مقایسه آن‌هاست. اگر اختلاف زیادی وجود داشته باشد، آن ویژگی در تفکیک اسپم‌ها مؤثرتر است.

```python
# Calculate mean of each feature in Spam vs Ham
spam_means = data[data['label']==1].mean()
ham_means  = data[data['label']==0].mean()

# Difference between Spam and Ham
feature_diff = (spam_means - ham_means).abs().sort_values(ascending=False)

# Show top 10 features
print(feature_diff.head(10))
```

📌 این خروجی نشان می‌دهد کدام ویژگی‌ها اختلاف بیشتری بین Spam و Ham دارند، بنابراین در پیش‌بینی اهمیت بالاتری دارند.

---

### 🔹 نمایش ویژگی‌های مهم با Bar Plot

```python
# Bar plot of top 10 important features
plt.figure(figsize=(10,5))
feature_diff.head(10).plot(kind='bar', color="purple")
plt.title("Top 10 Important Features (Difference Spam vs Ham)")
plt.ylabel("Absolute Mean Difference")
plt.xlabel("Features")
plt.show()
```

📌 این نمودار ویژگی‌هایی مثل **Free**، **Win**، یا **Offer** را به‌عنوان مهم‌ترین کلمات در تشخیص اسپم نمایش می‌دهد.

---

### 🔹 تحلیل دقیق‌تر با Boxplot

برای درک بهتر توزیع داده‌ها، چند ویژگی مهم را با **Boxplot** بررسی می‌کنیم.

```python
important_features = feature_diff.head(3).index  # سه ویژگی اول

plt.figure(figsize=(12,5))
for i, feat in enumerate(important_features, 1):
    plt.subplot(1,3,i)
    sns.boxplot(x="label", y=feat, data=data)
    plt.xticks([0,1], ['Ham','Spam'])
    plt.title(f"Boxplot of {feat}")
plt.tight_layout()
plt.show()
```

📌 این نمودارها نشان می‌دهند که چگونه مقادیر این ویژگی‌ها در اسپم‌ها به‌طور معناداری از ایمیل‌های عادی بیشتر یا کمتر هستند.

---

### 🔹 جمع‌بندی این صفحه

* یاد گرفتیم که انتخاب ویژگی‌های مهم باعث بهبود مدل می‌شود.
* با مقایسه میانگین ویژگی‌ها در Spam و Ham، توانستیم ویژگی‌های کلیدی را شناسایی کنیم.
* با استفاده از **Bar Plot** و **Boxplot**، نقش این ویژگی‌ها را به‌صورت بصری مشاهده کردیم.

در مرحله‌ی بعد، از همین ویژگی‌ها برای ساخت مدل **Naïve Bayes** استفاده می‌کنیم.

