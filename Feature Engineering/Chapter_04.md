# 📘 فصل چهارم: انتخاب ویژگی‌ها (Feature Selection)

🔹 هدف انتخاب ویژگی این است که از بین ۳۰ ویژگی دیتاست سرطان سینه، فقط آن‌هایی را نگه داریم که بیشترین نقش را در پیش‌بینی کلاس خروجی (Benign یا Malignant) دارند.

---

## 📊 ۱. انتخاب ویژگی بر اساس همبستگی

گاهی ویژگی‌ها اطلاعات مشابهی را منتقل می‌کنند. مثلاً **mean radius**، **mean perimeter** و **mean area** همگی تقریباً یکدیگر را تکرار می‌کنند.

```python
import numpy as np

# Calculate correlation
corr_matrix = df.drop("target", axis=1).corr().abs()

# Upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Select features with high correlation
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
to_drop
```

📌 نتیجه: برخی ویژگی‌ها مثل **mean perimeter** و **mean area** همبستگی بالای ۰.۹ دارند و می‌توانیم یکی از آن‌ها را حذف کنیم.

---

## 🌳 ۲. انتخاب ویژگی با استفاده از اهمیت ویژگی‌ها (Random Forest)

مدل‌های درختی مثل **Random Forest** به ما می‌گویند هر ویژگی چه مقدار در پیش‌بینی کلاس تأثیر دارد.

```python
from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
X = df.drop("target", axis=1)
y = df["target"]
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10)
```

📌 تفسیر:
ویژگی‌هایی مثل **worst concave points**، **worst perimeter** و **mean concavity** بیشترین اهمیت را دارند.

---

## 📊 نمایش اهمیت ویژگی‌ها با نمودار Barplot

```python
plt.figure(figsize=(10,6))
importances.sort_values(ascending=False).head(10).plot(kind="bar", color="teal")
plt.title("Top 10 Important Features (Random Forest)")
plt.ylabel("Importance Score")
plt.show()
```

📌 نتیجه:
می‌توانیم ببینیم که ویژگی‌های مربوط به **worst measurements** معمولاً اهمیت بالایی دارند و مدل آن‌ها را کلیدی تشخیص داده است.

---

## 🔁 ۳. انتخاب ویژگی با روش Recursive Feature Elimination (RFE)

RFE یک روش تکراری است که به تدریج ویژگی‌های کم‌اهمیت را حذف می‌کند.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Logistic Regression for RFE
model = LogisticRegression(max_iter=5000)
rfe = RFE(model, n_features_to_select=10)
fit = rfe.fit(X, y)

selected_features = X.columns[fit.support_]
selected_features
```

📌 خروجی:
لیست ۱۰ ویژگی منتخب بر اساس RFE. این ویژگی‌ها بیشترین تأثیر را در پیش‌بینی دارند.

---

## ✨ جمع‌بندی فصل

در این فصل یاد گرفتیم:

1. با استفاده از همبستگی، ویژگی‌های تکراری را حذف کنیم.
2. با Random Forest مهم‌ترین ویژگی‌ها را شناسایی کنیم.
3. با RFE مجموعه‌ای از ویژگی‌های بهینه برای مدل‌سازی انتخاب کنیم.
