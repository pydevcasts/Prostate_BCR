
## 📖 صفحه ۵: ساخت مدل KNN و ارزیابی اولیه

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 آماده‌سازی داده‌ها

قبل از ساخت مدل KNN باید داده‌ها را به **ویژگی‌ها (X)** و **برچسب‌ها (y)** تقسیم کنیم و سپس آموزش و تست را جدا کنیم.

```python
from sklearn.model_selection import train_test_split
# Separate features and target variable
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
# Split data into training and testing sets (70% train, 30% test)
# Use stratify=y to preserve class distribution in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
)
```

---

### 🔹 نرمال‌سازی داده‌ها

از آنجایی که KNN به مقیاس داده‌ها حساس است (مثلاً فاصله Glucose=150 با BMI=30 قابل مقایسه نیست)، باید داده‌ها را نرمال‌سازی کنیم.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

### 🔹 ساخت مدل KNN

حالا مدل KNN را با مقدار پیش‌فرض $k=5$ می‌سازیم:

```python
from sklearn.neighbors import KNeighborsClassifier
# Initialize KNN model with k=5
knn = KNeighborsClassifier(n_neighbors=5)
# Train the model on the training data
knn.fit(X_train, y_train)
# Make predictions on the test set
y_pred = knn.predict(X_test)
```
---

### 🔹 ارزیابی مدل

ابتدا دقت مدل را محاسبه می‌کنیم:

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

📌 خروجی (تقریبی):

* **دقت (Accuracy):** حدود 0.72
* **Confusion Matrix:** نشان می‌دهد چند نفر درست یا غلط پیش‌بینی شده‌اند.
* **Classification Report:** شامل Precision، Recall و F1-Score است.

---

### 🔹 تفسیر نتایج

* دقت مدل KNN در نسخه اولیه حدود **۷۲٪** است.
* مدل توانسته بیماران دیابتی را نسبتاً خوب شناسایی کند اما هنوز جای بهبود دارد.
* انتخاب **K مناسب** و همچنین **انتخاب ویژگی‌ها (Feature Selection)** می‌تواند کیفیت مدل را بالاتر ببرد.

