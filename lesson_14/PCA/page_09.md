

---

## 📖 صفحه ۹: استفاده از PCA در مدل‌های یادگیری ماشین

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 چرا PCA در یادگیری ماشین مفید است؟

مدل‌های یادگیری ماشین وقتی تعداد ویژگی‌ها زیاد باشد، ممکن است:

* دچار **Overfitting** شوند.
* سرعت آموزش پایین بیاید.
* همبستگی زیاد بین ویژگی‌ها باعث کاهش دقت شود.

PCA با کاهش بعد به ما کمک می‌کند:

1. داده‌ها سریع‌تر پردازش شوند.
2. نویز و همبستگی اضافی حذف شود.
3. مدل ساده‌تر ولی مؤثرتر باشد.

---

### 🔹 مقایسه مدل Logistic Regression با و بدون PCA

ابتدا مدل را روی داده اصلی (بدون کاهش بعد) تست می‌کنیم.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, data['target'], test_size=0.2, random_state=42)

# Train Logistic Regression without PCA
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print("Accuracy without PCA:", accuracy_score(y_test, y_pred))
```

---

سپس همان مدل را با داده‌های کاهش‌یافته توسط PCA (فقط ۳ مؤلفه اول) اجرا می‌کنیم:

```python
# Apply PCA with 3 components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Train-test split after PCA
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, data['target'], test_size=0.2, random_state=42)

# Train Logistic Regression with PCA
logreg_pca = LogisticRegression(max_iter=1000)
logreg_pca.fit(X_train_pca, y_train)
y_pred_pca = logreg_pca.predict(X_test_pca)

print("Accuracy with PCA (3 components):", accuracy_score(y_test, y_pred_pca))
```

---

### 🔹 تحلیل نتایج

📌 معمولاً روی دیتاست Wine:

* دقت مدل بدون PCA حدود **۹۸٪** است.
* دقت مدل با ۳ مؤلفه اول PCA حدود **۹۶٪** است.

یعنی با وجود کاهش بعد از ۱۳ ویژگی به ۳ ویژگی، همچنان دقت مدل تقریباً حفظ شده است و سرعت پردازش افزایش می‌یابد.

---

### 🔹 نتیجه این بخش

1. PCA ابزاری عالی برای کاهش بعد و جلوگیری از Overfitting است.
2. حتی با استفاده از ۲ یا ۳ مؤلفه، می‌توان دقت مدل‌ها را تقریباً حفظ کرد.
3. در پروژه‌های بزرگ، ترکیب PCA با الگوریتم‌های یادگیری ماشین یک روش بهینه‌سازی رایج است.

