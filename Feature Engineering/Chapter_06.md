# 📘 فصل ششم: پروژه نهایی – مقایسه عملکرد مدل‌ها

---

## 🎯 هدف این پروژه

ما می‌خوایم روی دیتاست **سرطان سینه** سه حالت رو تست کنیم:

۱. استفاده از **همه‌ی ویژگی‌ها**
۲. استفاده از **ویژگی‌های انتخاب‌شده (Feature Selection)**
۳. استفاده از **ویژگی‌های استخراج‌شده (PCA)**

و در هر حالت، دقت مدل رو با هم مقایسه کنیم.

---

## 📊 آماده‌سازی داده‌ها

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Split data
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 🔹 حالت اول: همه‌ی ویژگی‌ها

```python
# Logistic Regression with all features
model_all = LogisticRegression(max_iter=5000)
model_all.fit(X_train, y_train)
y_pred_all = model_all.predict(X_test)
acc_all = accuracy_score(y_test, y_pred_all)
acc_all
```

📌 دقت در این حالت معمولاً **۹۴-۹۶٪** خواهد بود.

---

## 🔹 حالت دوم: ویژگی‌های انتخاب‌شده (Feature Selection)

برای این بخش، مثلاً ۱۰ ویژگی مهم انتخاب‌شده با RFE یا Random Forest رو استفاده می‌کنیم:

```python
# Select top features (example: from Random Forest)
selected_features = ['worst concave points', 'worst perimeter', 
                     'mean concavity', 'worst radius', 'worst area',
                     'mean radius', 'mean perimeter', 'mean texture',
                     'mean smoothness', 'mean compactness']

X_train_fs = X_train[selected_features]
X_test_fs = X_test[selected_features]

# Train Logistic Regression
model_fs = LogisticRegression(max_iter=5000)
model_fs.fit(X_train_fs, y_train)
y_pred_fs = model_fs.predict(X_test_fs)
acc_fs = accuracy_score(y_test, y_pred_fs)
acc_fs
```

📌 دقت در این حالت اغلب **۹۵-۹۷٪** خواهد بود.

---

## 🔹 حالت سوم: استخراج ویژگی‌ها (PCA)

```python
# Apply PCA with 10 components
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train Logistic Regression
model_pca = LogisticRegression(max_iter=5000)
model_pca.fit(X_train_pca, y_train)
y_pred_pca = model_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)
acc_pca
```

📌 دقت در این حالت معمولاً **۹۳-۹۵٪** است.

---

## 📊 مقایسه نتایج

```python
# Plot accuracies
results = pd.DataFrame({
    "Approach": ["All Features", "Feature Selection", "PCA"],
    "Accuracy": [acc_all, acc_fs, acc_pca]
})

sns.barplot(x="Approach", y="Accuracy", data=results, palette="Set2")
plt.title("Comparison of Model Accuracy")
plt.ylim(0.9, 1.0)
plt.show()
```

📌 تفسیر نمودار:

* **Feature Selection** کمی بهتر از استفاده از همه‌ی ویژگی‌ها عمل کرده چون ویژگی‌های تکراری حذف شده‌اند.
* **PCA** هم نتایج خوبی دارد اما گاهی کمی پایین‌تر است، چون مؤلفه‌های فشرده شده اطلاعاتی را از دست می‌دهند.

---

## ✨ نتیجه‌گیری پروژه

1. استفاده از همه‌ی ویژگی‌ها همیشه بهترین گزینه نیست، چون داده‌های تکراری و نویزی وجود دارند.
2. **Feature Selection** باعث می‌شود مدل ساده‌تر، سریع‌تر و حتی دقیق‌تر شود.
3. **PCA** ابزاری عالی برای کاهش ابعاد و تجسم داده‌هاست، ولی همیشه دقت مدل را افزایش نمی‌دهد.

