## 📖 صفحه ۴: آموزش مدل درخت تصمیم

### 🔹 گام ۱: آماده‌سازی داده‌ها

برای آموزش مدل ابتدا باید داده‌ها را به دو بخش تقسیم کنیم: **آموزش (Train)** و **تست (Test)** تا بتوانیم دقت مدل را ارزیابی کنیم.

```python
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Features (X) and Target (y)
X = df.drop("species", axis=1)   # All 4 features
y = df["species"]                # Target (species)

# Split dataset into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
```

📌 توضیح:

* `train_test_split` داده‌ها را تقسیم می‌کند.
* ۸۰٪ داده‌ها برای آموزش و ۲۰٪ برای تست استفاده می‌شوند.

---

### 🔹 گام ۲: ساخت مدل درخت تصمیم

اکنون یک مدل درخت تصمیم ساده می‌سازیم.

```python
# Create Decision Tree Classifier
model = DecisionTreeClassifier(criterion="entropy", random_state=42)

# Train the model
model.fit(X_train, y_train)

# Accuracy on test set
accuracy = model.score(X_test, y_test)
print("Model Accuracy on Test Set:", accuracy)
```

📊 خروجی نشان خواهد داد که دقت مدل معمولاً بالای ۹۰٪ است (گاهی حتی ۱۰۰٪ چون دیتاست Iris ساده است).

---

### 🔹 گام ۳: اهمیت ویژگی‌ها (Feature Importance)

مدل درخت تصمیم به ما می‌گوید هر ویژگی چه میزان در تصمیم‌گیری تأثیر داشته است.

```python
# Feature importance
import numpy as np

feature_importance = model.feature_importances_
for name, importance in zip(X.columns, feature_importance):
    print(f"{name}: {importance:.4f}")

# Plot feature importance
plt.figure(figsize=(8,6))
sns.barplot(x=feature_importance, y=X.columns, palette="viridis")
plt.title("Feature Importance in Decision Tree", fontsize=14)
plt.show()
```

📊 **تفسیر:**

* معمولاً **Petal Length** و **Petal Width** بیشترین اهمیت را دارند.
* **Sepal Width** و **Sepal Length** اهمیت کمتری دارند.
* این یافته دقیقاً با تحلیل Heatmap و Boxplot صفحه قبل هم‌خوانی دارد.

---

### 🔹 گام ۴: ترسیم درخت تصمیم

یکی از زیبایی‌های درخت تصمیم این است که می‌توان آن را به‌صورت بصری رسم کرد.

```python
# Plot the trained decision tree
plt.figure(figsize=(16,10))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True, rounded=True)
plt.title("Decision Tree for Iris Dataset", fontsize=16)
plt.show()
```

🌳 درخت رسم‌شده نشان می‌دهد مدل چگونه ویژگی‌ها را به‌عنوان شرط انتخاب کرده است. مثلاً:

* اگر **Petal Length < 2.45** باشد → گل از گونه Setosa است.
* اگر بزرگ‌تر باشد، مدل وارد بررسی Petal Width می‌شود و بین Versicolor و Virginica تفکیک می‌کند.

---

### 🔹 گام ۵: پیش‌بینی روی داده‌های جدید

برای اطمینان از کارایی مدل، می‌توانیم چند نمونه جدید به مدل بدهیم:

```python
# Example new data
new_data = [[5.1, 3.5, 1.4, 0.2],   # Expected: Setosa
            [6.5, 2.8, 4.6, 1.5],   # Expected: Versicolor
            [7.2, 3.0, 5.8, 1.6]]   # Expected: Virginica

predictions = model.predict(new_data)
print("Predictions:", predictions)
```

---

### 🎯 نتیجه این صفحه

* مدل درخت تصمیم با معیار **Entropy** ساخته شد.
* دقت مدل روی داده‌های تست بسیار بالا بود.
* ویژگی‌های گلبرگ (Petal) بیشترین اهمیت را دارند.
* درخت تصمیم به صورت بصری رسم شد و مسیرهای تصمیم‌گیری مشخص شد.

