## 📖 صفحه ۷: بهبود مدل با استفاده از TF-IDF

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 مشکل CountVectorizer

در صفحات قبلی از **CountVectorizer** برای تبدیل متن به بردار عددی استفاده کردیم. در این روش هر کلمه صرفاً بر اساس **تعداد تکرار در متن** به یک ویژگی عددی تبدیل می‌شود.
اما این روش چند مشکل دارد:

* کلماتی مثل *the*، *is*، *and* تقریباً در همه‌ی متن‌ها زیاد تکرار می‌شوند ولی اطلاعات چندانی برای تشخیص اسپم ندارند.
* همه‌ی کلمات وزن یکسان دارند و هیچ تفاوتی بین کلمات خاص (مثل free, win, offer) و کلمات عمومی ایجاد نمی‌شود.

---

### 🔹 معرفی TF-IDF

برای رفع این مشکل از **TF-IDF (Term Frequency – Inverse Document Frequency)** استفاده می‌کنیم. این روش به هر کلمه بر اساس اهمیت آن در متن **وزن بیشتری** می‌دهد.

فرمول کلی:

$$
TF-IDF(t,d) = TF(t,d) \cdot IDF(t)
$$

* $TF(t,d)$: تعداد دفعاتی که کلمه $t$ در سند $d$ ظاهر شده است.
* $IDF(t)$: لگاریتم معکوس تعداد اسنادی که کلمه $t$ در آن‌ها وجود دارد.

به بیان ساده:

* اگر کلمه‌ای در یک متن زیاد تکرار شده باشد → امتیاز بالاتر.
* اگر همان کلمه در همه‌ی متن‌ها زیاد دیده شود → امتیازش کاهش پیدا می‌کند.

بنابراین کلماتی مثل *free* یا *win* که بیشتر در اسپم‌ها دیده می‌شوند وزن بیشتری می‌گیرند، در حالی که کلماتی مثل *the* تقریباً نادیده گرفته می‌شوند.

---

### 🔹 پیاده‌سازی در پایتون

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# استفاده از TF-IDF برای تبدیل متن‌ها به بردار
tfidf = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf.fit_transform(X)

# تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# ساخت مدل و آموزش
nb_model_tfidf = MultinomialNB()
nb_model_tfidf.fit(X_train, y_train)

# پیش‌بینی
y_pred_tfidf = nb_model_tfidf.predict(X_test)

# ارزیابی
print("Accuracy (TF-IDF):", accuracy_score(y_test, y_pred_tfidf))
print(classification_report(y_test, y_pred_tfidf, target_names=["Ham","Spam"]))
```

---

### 🔹 مقایسه CountVectorizer و TF-IDF

* **CountVectorizer**: همه کلمات را با وزن یکسان در نظر می‌گیرد.
* **TF-IDF**: کلمات خاص‌تر را برجسته می‌کند و کلمات عمومی را کاهش وزن می‌دهد.

📌 در عمل، مدل Naïve Bayes با TF-IDF معمولاً دقت بالاتری دارد (مثلاً Accuracy از ۹۵٪ به ۹۷٪ برسد).

---

### 🔹 نتیجه این مرحله

استفاده از **TF-IDF** باعث شد مدل حساسیت بیشتری نسبت به کلمات کلیدی مهم داشته باشد و خطاهای مربوط به کلمات عمومی کاهش یابد.
