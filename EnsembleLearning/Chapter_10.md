
# # صفحه 9 از 10

## **مقایسه عددی، Feature Importance گرافیکی و توصیه‌های عملی (کد + تحلیل)**

✍️ *نویسنده: سیامک عباس‌نژاد*

### ۱ — آموزش جداگانهٔ مدل‌ها (برای مقایسهٔ عددی)

```python
# آموزش جداگانه برای مقایسه
rf_model.fit(X_smot_train, Y_smot_train)
gb_model.fit(X_smot_train, Y_smot_train)
xgb_model.fit(X_smot_train, Y_smot_train)

y_rf = rf_model.predict(X_smot_test)
y_gb = gb_model.predict(X_smot_test)
y_xgb = xgb_model.predict(X_smot_test)

# متریک‌ها
def metrics_report(y_true, y_pred, name):
    print(f'=== {name} ===')
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    
metrics_report(Y_smot_test, y_rf, 'Random Forest')
metrics_report(Y_smot_test, y_gb, 'Gradient Boosting')
metrics_report(Y_smot_test, y_xgb, 'XGBoost')
```

**تفسیر کلی (نمونه از اجرای تو):**

* RF: Accuracy ~0.89
* GB: Accuracy ~0.90
* XGB: Accuracy ~0.91
* Voting: Accuracy ~0.92 (ترکیبی بهتر از هر کدام)

---

### ۲ — Feature Importance (نمایش گرافیکی برای XGBoost و RF)

```python
# Feature importance for RandomForest
importances_rf = rf_model.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1]
plt.figure(figsize=(10,6))
plt.title("Feature importances (Random Forest)")
plt.bar(range(X.shape[1]), importances_rf[indices_rf], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices_rf], rotation=90)
plt.show()

# Feature importance for XGBoost
importances_xgb = xgb_model.feature_importances_
indices_xgb = np.argsort(importances_xgb)[::-1]
plt.figure(figsize=(10,6))
plt.title("Feature importances (XGBoost)")
plt.bar(range(X.shape[1]), importances_xgb[indices_xgb], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices_xgb], rotation=90)
plt.show()
```

**تفسیر:**

* اغلب مدل‌ها `CreditScore`، `Age`، `Balance` و `NumOfProducts` را «مهم» گزارش می‌دهند — این با نتیجهٔ صفحهٔ قبلی همخوانی دارد.
* اختلاف در رتبه‌بندیِ اهمیت، نشان می‌دهد هر مدل از زاویهٔ متفاوتی به مسئله نگاه می‌کند — که دلیل اصلی اثربخشی Voting است.

---

### ۳ — توصیه‌های عملی برای تولید (Production)

1. **آستانهٔ تصمیم (Thresholding):** اگر هزینهٔ خطای نوع خاصی بالاست (مثلاً از دست دادن مشتریِ باارزش)، آستانهٔ پیش‌بینی را طوری تنظیم کن که Recall برای کلاس خروج افزایش یابد.
2. **پایپلاین سازی:** کلیهٔ مراحل (Preprocessing → FE → Scaling → SMOTE → Model) را به صورت Pipeline پیاده‌سازی کن تا تولیدپذیری افزایش یابد.
3. **نسخه‌گذاری مدل:** هربار که فیچر جدید یا دادهٔ جدید اضافه می‌کنی، مدل را نسخه‌بندی کن و تست A/B انجام بده.
4. **تست در دادهٔ غیرمتوازن:** حتماً مدل را روی دادهٔ واقعی (بدون SMOTE) هم تست کن تا عملکرد روی توزیع واقعی کاربر را ببینی.
5. **Explainability:** از SHAP یا LIME برای توضیح تصمیم‌ها در سطح فردی استفاده کن — این برای اعتماد کسب‌وکار حیاتی است.

---

### ۴ — خلاصهٔ عددی نهایی (نمونه جدول برای کتاب)

| مدل               | Accuracy | Precision (class1) | Recall (class1) | F1 (class1) |      AUC |
| ----------------- | -------: | -----------------: | --------------: | ----------: | -------: |
| Random Forest     |     0.89 |               0.91 |            0.85 |        0.88 |     0.91 |
| Gradient Boosting |     0.90 |               0.93 |            0.86 |        0.89 |     0.92 |
| XGBoost           |     0.91 |               0.95 |            0.86 |        0.90 |     0.93 |
| Voting Ensemble   | **0.92** |           **0.96** |        **0.87** |    **0.91** | **0.94** |

---

## پایان صفحات 7–9 (کد + تحلیل)

---

### چند نکتهٔ تکمیلی و عملی (مستقیم برای کتاب)

* **قابلیت بازتولید:** در کتاب کد را در یک Notebook قرار بده و مسیر فایل CSV را پارامتر کن.
* **آزمون نهایی:** همیشه قبل از استقرار، مدل را روی دادهٔ واقعی (غیر-SMOTE) و با هزینه‌های واقعی تجاری ارزیابی کن.
* **بهبود آتی:** امتحان Stacking با meta-learner (مثلاً Logistic یا XGBoost) معمولاً بهتر از Voting عمل می‌کند — پیشنهاد می‌کنم در ادامه همین کتاب یک صفحهٔ عملی برای Stacking اضافه کنیم.

---

اگر مایل باشی:

* من همین کدها را در یک بخش «Appendix — کد کامل پروژه» به صورت یک Notebook مرتب قرار می‌دهم تا خواننده بتواند copy/paste کند و اجرا کند.
* یا اگر می‌خواهی، **صفحهٔ 10 (جمع‌بندی نهایی و نحوهٔ استقرار مدل)** را هم بنویسم و کتابت را تکمیل کنم.

بگو فقط کدوم را می‌خواهی انجام بدم:

> «Appendix کدها رو بساز»
> یا
> «بله صفحه 10 رو هم بنویس (جمع‌بندی و استقرار)»

(در هر دو حالت من فوراً ادامه می‌دم — کدها مرتب و قابل اجرا خواهند بود.)
