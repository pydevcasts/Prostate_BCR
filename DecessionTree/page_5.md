## 📖 صفحه ۵: ارزیابی مدل و تحلیل نهایی پروژه Iris

### 🔹 گام ۱: ماتریس سردرگمی (Confusion Matrix)

یکی از بهترین ابزارها برای بررسی عملکرد مدل، **ماتریس سردرگمی** است. این ماتریس نشان می‌دهد چند نمونه به‌درستی یا به‌اشتباه طبقه‌بندی شده‌اند.

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Predictions on test set
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

📊 **تفسیر:**

* مقادیر روی قطر اصلی ماتریس تعداد پیش‌بینی‌های درست هستند.
* مقادیر خارج از قطر نشان می‌دهند که مدل اشتباه کرده است.
* در دیتاست Iris معمولاً همه یا تقریباً همه نمونه‌ها به‌درستی طبقه‌بندی می‌شوند.

---

### 🔹 گام ۲: معیارهای آماری (Precision, Recall, F1)

علاوه بر دقت کلی، می‌توانیم شاخص‌های دقیق‌تر ببینیم:

```python
# Classification report
report = classification_report(y_test, y_pred, target_names=model.classes_)
print(report)
```

📌 این گزارش شامل:

* **Precision:** درصد نمونه‌های درست در میان پیش‌بینی‌های یک کلاس
* **Recall:** درصد نمونه‌های درست شناسایی‌شده از کل واقعی‌ها
* **F1-score:** میانگین هماهنگ Precision و Recall

نتایج معمولاً بسیار بالا هستند (حدود ۱.۰).

---

### 🔹 گام ۳: نمایش توزیع پیش‌بینی‌ها

برای مقایسه توزیع واقعی و پیش‌بینی‌شده گونه‌ها:

```python
# Compare actual vs predicted
plt.figure(figsize=(8,5))
sns.countplot(x=y_test, palette="Set2", alpha=0.6, label="Actual")
sns.countplot(x=y_pred, palette="Set1", alpha=0.6, label="Predicted")
plt.legend(["Actual","Predicted"])
plt.title("Comparison of Actual vs Predicted Classes")
plt.show()
```

📊 **تفسیر:**

* ستون‌های Actual و Predicted تقریباً روی هم می‌افتند → نشان‌دهنده دقت بالا است.

---

### 🔹 گام ۴: مقایسه ویژگی‌های مهم

همان‌طور که در صفحه قبل دیدیم، ویژگی‌های گلبرگ (Petal Length و Petal Width) اهمیت بیشتری داشتند. حالا این را به صورت نمودار ترکیبی نشان می‌دهیم.

```python
# Jointplot to visualize important features
sns.jointplot(x="petal length (cm)", y="petal width (cm)", hue="species", data=df, height=6, palette="husl")
plt.show()
```

📊 **تفسیر:**

* گونه Setosa در گوشه‌ای مجزا قرار دارد (Petal کوچک‌تر).
* Versicolor و Virginica کمی همپوشانی دارند، اما باز هم قابل تفکیک هستند.

---

### 🔹 گام ۵: کاربردهای عملی درخت تصمیم

اکنون که دیدیم درخت تصمیم روی دیتاست Iris چگونه کار می‌کند، بیایید به دنیای واقعی نگاه کنیم:

* 🏥 **پزشکی:** تشخیص بیماری‌ها بر اساس علائم و آزمایش‌ها
* 💳 **بانکداری:** شناسایی مشتریان پرخطر یا پیش‌بینی کسانی که وام خود را بازپرداخت نمی‌کنند
* 🛍️ **بازاریابی:** تقسیم‌بندی مشتریان و پیشنهاد محصولات مناسب
* 🚦 **حمل‌ونقل:** پیش‌بینی ترافیک یا انتخاب مسیر بهینه

همان‌طور که می‌بینید، درخت تصمیم ابزاری ساده ولی بسیار پرقدرت است که تقریباً در هر حوزه‌ای می‌تواند مورد استفاده قرار بگیرد.

---

### 🎯 نتیجه نهایی

* مدل درخت تصمیم روی دیتاست Iris توانست گونه‌ها را با دقت بسیار بالا طبقه‌بندی کند.
* Heatmap و Boxplot نشان دادند که ویژگی‌های گلبرگ بیشترین اهمیت را دارند.
* Confusion Matrix و گزارش آماری دقت عالی مدل را تایید کردند.
* کاربردهای درخت تصمیم فراتر از مثال‌های ساده است و در صنایع مختلف از پزشکی تا بازاریابی استفاده می‌شود.
