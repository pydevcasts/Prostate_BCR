## 📖 صفحه ۱۵: اهمیت ویژگی‌ها در Random Forest

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 چرا Feature Importance مهم است؟

* در مسائل پزشکی، علاوه بر پیش‌بینی دقیق، **توضیح‌پذیری (Interpretability)** مدل اهمیت زیادی دارد.
* پزشکان باید بدانند **کدام ویژگی‌های بیولوژیکی یا اندازه‌گیری‌ها** بیشترین نقش را در تشخیص دارند.
* Random Forest این امکان را فراهم می‌کند که اهمیت ویژگی‌ها را مستقیماً محاسبه کنیم.

---

### 🔹 کدنویسی: استخراج Feature Importance

```python
# Train Random Forest on full dataset
rf_model.fit(X_scaled, y)

# Extract feature importances
importances = rf_model.feature_importances_

# Create DataFrame for visualization
feat_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(feat_importances.head(10))  # Top 10 features
```

---

### 🔹 نمایش با Barplot

```python
plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=feat_importances.head(15), palette="viridis")
plt.title("Top 15 Feature Importances in Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
```

---

### 🔹 تفسیر نتایج (مثال فرضی)

* ویژگی‌های مثل **worst radius**، **mean concave points** و **worst area** بیشترین اهمیت را داشته‌اند.
* ویژگی‌های مثل **fractal dimension** یا **symmetry** اهمیت کمتری داشته‌اند.

📌 این نتیجه نشان می‌دهد که **شکل و اندازه سلول‌ها** (radius, area, concavity) کلیدی‌ترین نقش را در تشخیص سرطان دارند.

---

### 🔹 نتیجه‌گیری

* Random Forest علاوه بر دقت بالا، قابلیت توضیح‌پذیری هم دارد.
* تحلیل Feature Importance به پزشکان کمک می‌کند بفهمند **کدام ویژگی‌های بیولوژیکی حساس‌تر هستند**.
* این نتایج می‌تواند پایه‌ای برای تحقیقات پزشکی عمیق‌تر باشد.

