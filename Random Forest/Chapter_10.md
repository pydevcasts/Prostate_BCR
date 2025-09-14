## 📖 صفحه ۱۰: تحلیل Feature Importance در Random Forest

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 چرا Feature Importance مهم است؟

مدل‌های درختی مثل **Random Forest** این امکان را دارند که مشخص کنند **کدام ویژگی‌ها بیشترین نقش** را در پیش‌بینی دارند.
این تحلیل کمک می‌کند:

1. بفهمیم کدام ویژگی‌ها کلیدی هستند.
2. ویژگی‌های کم‌اهمیت را حذف کنیم (Feature Selection).
3. مدل‌های ساده‌تر و سریع‌تر بسازیم.

---

### 🔹 استخراج Feature Importance

```python
# Train Random Forest on full dataset
rf_model.fit(X_scaled, y)

# Get feature importances
importances = rf_model.feature_importances_

# Create a DataFrame
feat_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)
```

---

### 🔹 نمایش Feature Importance با Barplot

```python
plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=feat_importances, palette="viridis")
plt.title("Feature Importance in Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
```

---

### 🔹 تفسیر نتایج (مثال)

فرض کنید خروجی نشان دهد:

* **mean concave points** و **worst radius** بالاترین اهمیت را دارند.
* ویژگی‌هایی مثل **mean symmetry** یا **fractal dimension** اهمیت کمتری دارند.

📌 این یعنی مدل بیشترین تصمیماتش را بر اساس الگوهای هندسی مربوط به **شکل و اندازه سلول‌ها** می‌گیرد.

---

### 🔹 کاربرد این تحلیل

* ویژگی‌های با اهمیت بالا می‌توانند در تحقیقات پزشکی مورد توجه قرار گیرند.
* ویژگی‌های با اهمیت پایین شاید برای ساده‌تر کردن مدل حذف شوند.
* با ترکیب Feature Importance و روش‌های دیگر (مثل PCA یا LASSO) می‌توان بهترین مجموعه ویژگی‌ها را انتخاب کرد.

---

### 🔹 مقدمه برای ادامه

در صفحه بعد می‌توانیم:

* مقایسه Random Forest با مدل‌های دیگر (Logistic Regression, KNN, SVM).
* استفاده از Cross Validation برای همه مدل‌ها.
* نمایش نتایج مقایسه با نمودار (Barplot یا Boxplot).

