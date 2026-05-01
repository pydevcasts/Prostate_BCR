
## 🧩 فصل ۸: ایجاد ویژگی‌های جدید (Feature Construction & Generation)

### 📄 صفحه ۳ از ۵ — ایجاد ویژگی‌های متنی (Text Feature Engineering)

✍️ *نویسنده: سیامک عباس‌نژاد*
🌐 *[https://github.com/pydevcasts](https://github.com/pydevcasts)*

---

### 🎯 هدف این صفحه

در این بخش یاد می‌گیری:
✅ چطور داده‌های متنی رو به ویژگی‌های عددی تبدیل کنیم
✅ روش‌های Count Vector، TF-IDF و Word Embedding
✅ پیش‌پردازش متن (حذف stopwords، کوچک‌سازی حروف و ...)
✅ اجرای کامل با پایتون و تحلیل خروجی‌ها

---

## 🧠 ۱. چرا متن باید به عدد تبدیل بشه؟

مدل‌های یادگیری ماشین فقط **اعداد** رو می‌فهمن 🧮
پس وقتی داده‌ها متنی هستن (مثل کامنت‌ها، پست‌ها، ایمیل‌ها و نظرات کاربران)،
باید اون متن‌ها رو به **ویژگی‌های عددی معنی‌دار** تبدیل کنیم 💬➡️🔢

---

## ✨ ۲. پیش‌پردازش متن (Text Preprocessing)

قبل از تبدیل، باید متن رو تمیز کنیم.
چند مرحله‌ی پایه‌ای و ضروری:

| مرحله                  | توضیح              | مثال               |
| :--------------------- | :----------------- | :----------------- |
| lowercase              | تبدیل به حروف کوچک | "Hello" → "hello"  |
| remove punctuation     | حذف علائم نگارشی   | "hi!!!" → "hi"     |
| remove stopwords       | حذف کلمات بی‌اهمیت | "the", "and", "is" |
| stemming/lemmatization | ریشه‌یابی لغات     | "running" → "run"  |

---

### 💻 مثال پایتون — پاکسازی متن

```python
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

data = pd.DataFrame({
    'text': [
        "I love machine learning!",
        "Machine Learning is amazing.",
        "Deep learning builds on machine learning."
    ]
})

# پاکسازی متن
def clean_text(text):
    text = text.lower()                       # حروف کوچک
    text = re.sub(r'[^a-z\s]', '', text)      # حذف علائم
    tokens = text.split()
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    return " ".join(tokens)

data['clean_text'] = data['text'].apply(clean_text)
print(data)
```

📊 **خروجی:**

```
                              text                         clean_text
0       I love machine learning!               love machine learning
1      Machine Learning is amazing.                 machine learning amazing
2  Deep learning builds on machine learning.  deep learning builds machine learning
```

✅ حالا متن آماده است تا به ویژگی‌های عددی تبدیل بشه.

---

## 🔹 ۳. شمارش کلمات (Count Vectorization)

ساده‌ترین روش: بشمار چند بار هر کلمه در هر جمله اومده 🧾

### 💻 مثال پایتون

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['clean_text'])

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

📊 **خروجی:**

```
['amazing' 'builds' 'deep' 'learning' 'love' 'machine']
[[0 0 0 1 1 1]
 [1 0 0 1 0 1]
 [0 1 1 2 0 1]]
```

✅ اینجا هر جمله با یه آرایه‌ی عددی نمایش داده می‌شه که
تعداد تکرار هر کلمه رو نشون می‌ده.

---

## 🔹 ۴. فراوانی واژه – معکوس فراوانی سند (TF-IDF)
#### Term Frequency – Inverse Document Frequency

روش TF-IDF به جای شمردن ساده، بررسی می‌کنه **کدوم کلمه مهم‌تره** 📈

$$
TF\text{-}IDF = TF(word) × \log\frac{N}{DF(word)}
$$

که در اون:

* **TF(word):** تعداد تکرار کلمه در متن
* **DF(word):** تعداد متونی که اون کلمه در اون‌ها ظاهر شده
* **N:** تعداد کل متون

---

### 💻 مثال پایتون

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(data['clean_text'])

print(tfidf.get_feature_names_out())
print(X_tfidf.toarray())
```

📊 **خروجی (تقریبی):**

```
['amazing' 'builds' 'deep' 'learning' 'love' 'machine']
[[0.000 0.000 0.000 0.577 0.577 0.577]
 [0.707 0.000 0.000 0.408 0.000 0.577]
 [0.000 0.577 0.577 0.577 0.000 0.288]]
```

✅ TF-IDF به کلماتی که خاص‌ترند وزن بیشتری می‌ده.

---

## 🔹 ۵. بردارهای معنایی (Word Embeddings)

در روش‌های جدیدتر (مثل Word2Vec یا BERT)،
هر کلمه به یه **بردار عددی چندبعدی** تبدیل می‌شه که معنای اون کلمه رو در خودش داره 🤖

| کلمه  | بردار نمونه (۵ بعدی)            |
| :---- | :------------------------------ |
| "cat" | [0.21, 0.43, -0.12, 0.87, 0.09] |
| "dog" | [0.20, 0.45, -0.11, 0.88, 0.07] |

می‌بینی که "cat" و "dog" بردارهای نزدیک به هم دارن چون معنی مشابه دارن 🐶🐱

---

### 💻 مثال ساده با Word2Vec

```python
from gensim.models import Word2Vec

sentences = [s.split() for s in data['clean_text']]
model = Word2Vec(sentences, vector_size=5, window=3, min_count=1, sg=1)

print(model.wv['machine'])
```

📊 **خروجی نمونه:**

```
[ 0.032 -0.078  0.104  0.216  0.048]
```

✅ حالا هر کلمه برداری داره که مفهومش رو در خودش نگه می‌داره.

---

### 💬 گفت‌وگوی استاد و دانشجو

👩‍💻 استاد، TF-IDF بهتره یا Word2Vec؟
👨‍🏫 بستگی داره —
TF-IDF برای داده‌های کوچیک و مدل‌های کلاسیک عالیه،
ولی Word2Vec یا BERT برای داده‌های بزرگ و مدل‌های عمیق بهترن 🧠

---

### 🎨 تصویر پیشنهادی

> تصویری که نشان دهد:
> متن → پاکسازی → Count Vector → TF-IDF → Word Embedding
> و در هر مرحله داده‌ها از حالت متن به حالت عددی و سپس معنایی تبدیل می‌شوند.

---

### 🧩 تمرین

۱️⃣ مجموعه‌ای از ۵ جمله‌ی کوتاه بساز.
۲️⃣ ابتدا با CountVectorizer و سپس با TfidfVectorizer آن‌ها را به بردار عددی تبدیل کن.
۳️⃣ تفاوت وزن‌دهی بین دو روش را مقایسه کن.

---

### ❓ پرسش چهارگزینه‌ای

کدام روش وزن بیشتری به کلمات خاص‌تر و کمتر تکرارشده می‌دهد؟

A) Count Vectorizer

B) Word2Vec

C) TF-IDF ✅

D) One-Hot Encoding

