## 📖 صفحه ۴: تحلیل همبستگی ویژگی‌ها با Heatmap

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 چرا همبستگی مهم است؟

در دیتاست‌هایی با تعداد ویژگی‌های زیاد (مثل Breast Cancer با ۳۰ ویژگی)، بررسی **ارتباط بین ویژگی‌ها** اهمیت بالایی دارد:

* ویژگی‌هایی که همبستگی بالایی دارند اطلاعات مشابهی ارائه می‌دهند.
* وجود همبستگی زیاد ممکن است باعث **Redundancy** یا **Multicollinearity** شود.
* Heatmap بهترین ابزار برای دیدن این روابط به صورت تصویری است.

---

### 🔹 محاسبه ماتریس همبستگی

```python
# Calculate correlation matrix
corr = df.corr()

# Plot heatmap
plt.figure(figsize=(14,10))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap of Breast Cancer Features")
plt.show()
```

📊 نتیجه:

* رنگ‌های قرمز پررنگ نشان‌دهنده‌ی همبستگی مثبت قوی هستند.
* رنگ‌های آبی پررنگ نشان‌دهنده‌ی همبستگی منفی قوی هستند.

---

### 🔹 نمونه‌ای از همبستگی‌های قوی

در این دیتاست:

* ویژگی‌های **mean radius**, **mean perimeter**, و **mean area** همبستگی بسیار قوی دارند (نزدیک به 0.99).
* ویژگی **mean compactness** و **mean concavity** نیز همبستگی بالایی نشان می‌دهند.
* برخی ویژگی‌ها مثل **mean fractal dimension** همبستگی کمی با بقیه دارند.

---

### 🔹 نمایش Heatmap برای همبستگی با Target

برای اینکه بفهمیم کدام ویژگی‌ها بیشترین ارتباط را با برچسب (target) دارند:

```python
# Correlation with target
corr_target = df.corr()['target'].sort_values(ascending=False)

# Show top correlated features
print(corr_target.head(10))
```

📊 نتایج نمونه:

* **worst perimeter** → همبستگی مثبت قوی با کلاس Malignant
* **mean concavity** → همبستگی مثبت قوی
* **mean radius** → همبستگی مثبت قوی
* **mean smoothness** → همبستگی ضعیف‌تر

---

### 🔹 نتیجه‌گیری

Heatmap نشان داد که:

* برخی ویژگی‌ها تقریباً اطلاعات تکراری دارند (Redundancy بالا).
* ویژگی‌هایی مثل **mean radius** و **worst perimeter** بیشترین تأثیر را در تشخیص Malignant vs Benign دارند.
* بعضی ویژگی‌ها اهمیت کمتری دارند و ممکن است در **Feature Selection** حذف شوند.

