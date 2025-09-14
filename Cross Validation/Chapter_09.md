## 📖 صفحه ۹: مقایسه مدل‌ها با معیارهای مختلف (Precision، Recall، F1-Score)

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 چرا فقط Accuracy کافی نیست؟

دقت (Accuracy) معیار خوبی است، اما همیشه تصویر کاملی از عملکرد مدل نمی‌دهد. به‌ویژه وقتی کلاس‌ها نامتوازن باشند. برای تحلیل بهتر از معیارهای زیر استفاده می‌کنیم:

* **Precision (دقت مثبت‌ها)**

$$
Precision = \frac{TP}{TP + FP}
$$

* **Recall (بازیابی یا حساسیت)**

$$
Recall = \frac{TP}{TP + FN}
$$

* **F1-Score (میانگین موزون Precision و Recall)**

$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

---

### 🔹 پیاده‌سازی و محاسبه معیارها برای هر مدل

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='linear')
}

reports = {}
conf_matrices = {}

for name, model in models.items():
    y_pred = cross_val_predict(model, X_scaled, y, cv=5)
    reports[name] = classification_report(y, y_pred, output_dict=True)
    conf_matrices[name] = confusion_matrix(y, y_pred)
```

---

### 🔹 نمایش نتایج در قالب جدول

```python
df_reports = pd.DataFrame({
    model: {
        "Precision": reports[model]["weighted avg"]["precision"],
        "Recall": reports[model]["weighted avg"]["recall"],
        "F1-Score": reports[model]["weighted avg"]["f1-score"]
    }
    for model in reports
}).T

print(df_reports)
```

📊 نمونه خروجی:

| Model               | Precision | Recall | F1-Score |
| ------------------- | --------- | ------ | -------- |
| Logistic Regression | 0.96      | 0.96   | 0.96     |
| KNN                 | 0.95      | 0.94   | 0.94     |
| SVM                 | 0.97      | 0.97   | 0.97     |

---

### 🔹 نمایش بصری با Barplot

```python
df_reports.plot(kind="bar", figsize=(10,6))
plt.title("Comparison of Models (Precision, Recall, F1-Score)")
plt.ylabel("Score")
plt.ylim(0.9, 1.0)
plt.grid(axis="y")
plt.show()
```

📌 این نمودار به‌وضوح نشان می‌دهد که **SVM در همه معیارها کمی بهتر از Logistic Regression و KNN است.**

---

### 🔹 تحلیل Confusion Matrix با Heatmap

```python
fig, axes = plt.subplots(1, 3, figsize=(18,5))

for i, (name, cm) in enumerate(conf_matrices.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
    axes[i].set_title(f"{name} Confusion Matrix")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

plt.show()
```

📊 در Confusion Matrix مشخص می‌شود هر مدل در تفکیک کلاس‌ها چقدر دقیق عمل کرده است.

---

### 🔹 جمع‌بندی صفحه ۹

* معیارهای پیشرفته‌تر (Precision، Recall، F1-Score) برای مقایسه مدل‌ها استفاده شد.
* **SVM بهترین عملکرد** را داشت، Logistic Regression در رتبه دوم و KNN کمی ضعیف‌تر.
* با استفاده از Heatmap و Barplot نتایج را به‌صورت تصویری بررسی کردیم.

