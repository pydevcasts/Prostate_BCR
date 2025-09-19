## 📖 صفحه ۸: تحلیل ویژگی‌ها و مهم‌ترین کلمات در تشخیص اسپم

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 چرا تحلیل ویژگی‌ها مهم است؟

تا اینجا دیدیم که مدل Naïve Bayes با استفاده از **TF-IDF** عملکرد خوبی در تشخیص اسپم دارد.
اما برای درک بهتر رفتار مدل باید بررسی کنیم **کدام کلمات بیشترین نقش در تصمیم‌گیری دارند**. این تحلیل نه تنها به ما کمک می‌کند که مدل را بهتر بشناسیم، بلکه به دانشجویان درک عمیق‌تری از رابطه بین داده و الگوریتم می‌دهد.

---

### 🔹 استخراج وزن ویژگی‌ها از مدل

در مدل Multinomial Naïve Bayes، ضرایب مدل را می‌توان از ویژگی‌های `coef_` استخراج کرد.
این ضرایب نشان می‌دهند هر کلمه چقدر در اسپم یا نان‌اسپم بودن مؤثر است.

```python
import numpy as np

# گرفتن لیست کلمات از بردارساز TF-IDF
feature_names = tfidf.get_feature_names_out()

# گرفتن ضرایب کلاس‌ها (log probability)
spam_coefficients = nb_model_tfidf.coef_[0]

# مرتب‌سازی کلمات بر اساس بیشترین وزن
top_indices = np.argsort(spam_coefficients)[-20:]
top_features = feature_names[top_indices]
top_values = spam_coefficients[top_indices]

# نمایش مهم‌ترین کلمات
for word, value in zip(top_features, top_values):
    print(word, ":", value)
```

---

### 🔹 نمایش گرافیکی با Bar Plot

برای درک بهتر، مهم‌ترین کلمات مؤثر در اسپم بودن را با نمودار میله‌ای نمایش می‌دهیم:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.barh(top_features, top_values, color="red")
plt.xlabel("Weight (log probability)")
plt.title("Top Words Contributing to Spam Detection")
plt.show()
```

📌 این نمودار به وضوح نشان می‌دهد کلماتی مثل:

* **free**
* **win**
* **offer**
* **click**
* **buy**

در ایمیل‌های اسپم وزن بیشتری دارند و عامل اصلی در تصمیم‌گیری مدل هستند.

---

### 🔹 تفسیر نتایج

* کلماتی که بیشتر در ایمیل‌های تبلیغاتی و اسپم دیده می‌شوند (مثل free, win, offer) در صدر لیست قرار گرفتند.
* کلمات عمومی و بی‌اهمیت مثل *the* یا *and* به دلیل وزن پایین در TF-IDF تقریباً نادیده گرفته شدند.
* این تحلیل نشان می‌دهد مدل علاوه بر دقت بالا، **قابل تفسیر** نیز هست، و ما می‌توانیم بفهمیم چرا یک ایمیل به‌عنوان اسپم برچسب خورده است.

