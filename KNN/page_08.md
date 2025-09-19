## 📖 صفحه ۸: تحلیل بصری ویژگی‌های مهم

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 اهمیت تحلیل بصری ویژگی‌ها

تحلیل عددی کافی نیست؛ برای درک بهتر اثر هر ویژگی باید آن را به صورت **نمودار** نمایش دهیم. این کار کمک می‌کند تا ببینیم کدام ویژگی‌ها در تمایز بیماران دیابتی و سالم نقش بیشتری دارند.

---

### 🔹 نمودار میله‌ای (Bar Plot) برای اهمیت ویژگی‌ها

ابتدا اهمیت ویژگی‌ها بر اساس امتیاز χ² در **SelectKBest** را نمایش می‌دهیم:

```python
import numpy as np
import matplotlib.pyplot as plt

# Get scores from SelectKBest
scores = selector.scores_

# Plot bar chart
plt.figure(figsize=(10,6))
plt.bar(X.columns, scores)
plt.xticks(rotation=45)
plt.ylabel("Chi2 Score")
plt.title("Feature Importance Based on Chi2 Test")
plt.show()
```

📌 نتیجه: معمولاً **Glucose** بالاترین امتیاز را دارد، سپس **BMI** و **Age**.

---

### 🔹 Boxplot برای مقایسه توزیع ویژگی‌ها

یکی از بهترین ابزارها برای بررسی تفاوت توزیع ویژگی‌ها بین بیماران دیابتی و غیردیابتی، **Boxplot** است.

```python
import seaborn as sns

plt.figure(figsize=(8,6))
sns.boxplot(x="Outcome", y="Glucose", data=data)
plt.title("Distribution of Glucose by Diabetes Outcome")
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x="Outcome", y="BMI", data=data)
plt.title("Distribution of BMI by Diabetes Outcome")
plt.show()
```

📌 تحلیل:

* بیماران دیابتی به طور متوسط **Glucose بالاتری** دارند.
* در ویژگی **BMI** نیز افراد دیابتی معمولاً مقادیر بالاتری نسبت به افراد سالم دارند.
* Boxplot همچنین نقاط پرت (Outliers) را به صورت دایره نمایش می‌دهد.

---

### 🔹 نتیجه‌گیری از تحلیل بصری

* ویژگی‌های **Glucose و BMI** بیشترین تأثیر را در تشخیص دیابت دارند.
* نمایش بصری به دانشجویان کمک می‌کند درک شهودی‌تری از داده‌ها پیدا کنند.
* این تحلیل‌ها نشان می‌دهد چرا انتخاب درست ویژگی‌ها باعث بهبود عملکرد مدل می‌شود.

---
