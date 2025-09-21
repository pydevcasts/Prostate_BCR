## 📖 صفحه ۷: انتخاب ویژگی‌های مؤثر (Feature Selection)

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 چرا Feature Selection مهم است؟

الگوریتم KNN به دلیل اینکه بر اساس فاصله کار می‌کند، نسبت به ویژگی‌های غیرضروری یا ویژگی‌هایی که مقیاس متفاوت دارند حساس است.

* وجود ویژگی‌های غیرمؤثر می‌تواند فاصله‌ها را گمراه کند.
* انتخاب ویژگی‌های مؤثر باعث **کاهش بعد داده‌ها** و افزایش سرعت و دقت مدل می‌شود.

---

### 🔹 استفاده از SelectKBest

یکی از روش‌های ساده برای انتخاب ویژگی‌ها، استفاده از **SelectKBest** است که بر اساس تست آماری (مثل χ²) ویژگی‌های مهم‌تر را انتخاب می‌کند.

```python
from sklearn.feature_selection import SelectKBest, chi2

# Apply SelectKBest with chi2 score function
selector = SelectKBest(score_func=chi2, k=5)
X_new = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()]

print("Selected Features:", selected_features)
```

📌 معمولاً ویژگی‌های **Glucose، BMI، Age** و گاهی **Pregnancies یا Insulin** به عنوان مؤثرترین انتخاب می‌شوند.

---

### 🔹 ساخت مدل KNN با ویژگی‌های منتخب

```python
# Split new dataset with selected features
X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(
    X[selected_features], y, test_size=0.3, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_fs = scaler.fit_transform(X_train_fs)
X_test_fs = scaler.transform(X_test_fs)

# Train KNN with selected features
knn_fs = KNeighborsClassifier(n_neighbors=9)  # chosen from previous step
knn_fs.fit(X_train_fs, y_train_fs)
y_pred_fs = knn_fs.predict(X_test_fs)

# Evaluate
print("Accuracy with Feature Selection:", accuracy_score(y_test_fs, y_pred_fs))
print("Confusion Matrix:\n", confusion_matrix(y_test_fs, y_pred_fs))
print("Classification Report:\n", classification_report(y_test_fs, y_pred_fs))
```

---

### 🔹 مقایسه با مدل قبلی

* دقت مدل بعد از انتخاب ویژگی‌ها معمولاً کمی بالاتر می‌رود (مثلاً از 73٪ به 76٪).
* علاوه بر دقت، مدل سبک‌تر و سریع‌تر اجرا می‌شود.
* این نشان می‌دهد که حذف ویژگی‌های کم‌اهمیت (مثل SkinThickness یا DiabetesPedigreeFunction) به نفع عملکرد مدل است.

