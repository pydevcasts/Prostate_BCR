## 📖 صفحه ۵: ساخت مدل Naïve Bayes و ارزیابی اولیه

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 آماده‌سازی داده‌ها

قبل از ساخت مدل، باید داده‌ها را به دو بخش تقسیم کنیم:

* **داده‌های آموزش (Train)** برای یادگیری مدل
* **داده‌های تست (Test)** برای ارزیابی عملکرد مدل روی داده‌های جدید

همچنین معمولاً برای داده‌های متنی، از روش‌های بردارسازی مثل **CountVectorizer** یا **TF-IDF** استفاده می‌کنیم. این روش‌ها متن را به یک ماتریس عددی تبدیل می‌کنند که مدل بتواند با آن کار کند.

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# تقسیم داده‌ها به متن (X) و برچسب‌ها (y)
X = data['text']      # ستون متن ایمیل
y = data['label']     # ستون برچسب (Spam=1, Ham=0)

# بردارسازی متن
vectorizer = CountVectorizer(stop_words='english')
X_vectors = vectorizer.fit_transform(X)

# تقسیم به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(
    X_vectors, y, test_size=0.2, random_state=42
)
```

---

### 🔹 آموزش مدل Naïve Bayes

برای متن، معمولاً از **Multinomial Naïve Bayes** استفاده می‌شود.

```python
from sklearn.naive_bayes import MultinomialNB

# ایجاد مدل
nb_model = MultinomialNB()

# آموزش مدل روی داده‌های آموزشی
nb_model.fit(X_train, y_train)
```

---

### 🔹 پیش‌بینی روی داده‌های تست

```python
# پیش‌بینی روی داده‌های تست
y_pred = nb_model.predict(X_test)
```

---

### 🔹 ارزیابی عملکرد مدل

برای ارزیابی، از معیارهای زیر استفاده می‌کنیم:

* **Accuracy** (دقت کلی)
* **Confusion Matrix** (ماتریس آشفتگی)
* **Classification Report** (Precision، Recall، F1-score)

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# دقت
print("Accuracy:", accuracy_score(y_test, y_pred))

# ماتریس آشفتگی
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["Ham","Spam"],
            yticklabels=["Ham","Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# گزارش دسته‌بندی
print(classification_report(y_test, y_pred, target_names=["Ham","Spam"]))
```

---

### 🔹 جمع‌بندی این مرحله

* داده‌ها را به **آموزش و تست** تقسیم کردیم.
* مدل **Naïve Bayes** را روی داده‌های آموزش یاد گرفتیم.
* با استفاده از داده‌های تست، عملکرد مدل را با معیارهای مهم ارزیابی کردیم.

در صفحه‌ی بعد، به **تحلیل دقیق‌تر نتایج** می‌پردازیم:

* بررسی اینکه مدل کدام نوع خطاها را بیشتر انجام می‌دهد.
* تحلیل اهمیت Precision و Recall در کاربرد واقعی.



