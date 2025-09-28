
# 📖 فصل ۹: ساخت مدل ترکیبی (ویژگی‌های متنی + ویژگی‌های عددی)

### 🔹 ترکیب ویژگی‌های متنی و عددی

تا اینجا ما داشتیم:

* **ویژگی‌های متنی** با استفاده از **TF-IDF** یا **CountVectorizer**
* **ویژگی‌های عددی** مثل طول پیام، تعداد حروف بزرگ، تعداد علامت تعجب و ...

حالا این دو نوع ویژگی را با هم ترکیب می‌کنیم تا عملکرد مدل بهبود پیدا کند.

```python
from scipy.sparse import hstack
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Selecting numeric features
numeric_features = df[['email_length', 'uppercase_count', 
                       'exclamation_count', 'digit_count',
                       'has_free', 'has_win', 'has_urgent', 'has_call']]

# Converting numeric features to sparse matrix for compatibility with TF-IDF
numeric_sparse = sparse.csr_matrix(numeric_features.values)

# Combining text features (TF-IDF) with numeric features
X_combined = hstack([X_tfidf, numeric_sparse])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)

# Training the Naive Bayes model
nb_model_combined = MultinomialNB()
nb_model_combined.fit(X_train, y_train)

# Prediction
y_pred_combined = nb_model_combined.predict(X_test)

# Evaluation
print("Accuracy (Hybrid Model):", accuracy_score(y_test, y_pred_combined))
print(classification_report(y_test, y_pred_combined, target_names=["Ham", "Spam"]))
```

📊 **نتیجه:**

* مدل ترکیبی که هم از ویژگی‌های متنی و هم از ویژگی‌های عددی استفاده می‌کند، معمولاً عملکرد **بهتری** نسبت به استفاده‌ی تنها از متن دارد.
* ویژگی‌های عددی مثل **تعداد حروف بزرگ، طول پیام و کلمات کلیدی اسپم** قدرت پیش‌بینی مدل را بیشتر می‌کنند.

---
