

📖 کتابخانه NLTK چیست؟

NLTK (Natural Language Toolkit) یکی از پرکاربردترین کتابخانه‌های پایتون برای پردازش زبان طبیعی (NLP) است.
این کتابخانه مجموعه‌ای از ابزارها، پیکره‌ها (دیتاست‌ها) و امکانات برای کار با داده‌های متنی (زبان انسانی) فراهم می‌کند.

کاربردهای اصلی NLTK:
	•	پیش‌پردازش متن (توکنیزه‌کردن، ریشه‌یابی، لماتیزه‌کردن و …)
	•	تحلیل زبان‌شناختی (برچسب‌گذاری اجزای کلام، تجزیه، شناسایی موجودیت‌ها)
	•	ساخت برنامه‌های NLP (چت‌بات، تحلیل احساسات، دسته‌بندی متن و …)
---

⚙️ ویژگی‌های کلیدی NLTK
	1.	ابزارهای پردازش متن
	•	توکنیزه‌کردن → تقسیم متن به کلمات یا جملات
	•	ریشه‌یابی (Stemming) → برگرداندن کلمات به ریشه (مثل: “running” → “run”)
	•	لماتیزه‌کردن → نسخه دقیق‌تر ریشه‌یابی که با توجه به معنا و فرهنگ لغت عمل می‌کند (مثل: “better” → “good”)
	•	حذف کلمات ایست (Stopwords) → حذف کلماتی مثل “است”، “و”، “که”
	•	برچسب‌گذاری اجزای کلام (POS Tagging) → تشخیص نقش کلمات (اسم، فعل، صفت و …)
	•	شناسایی موجودیت‌های نامدار (NER) → استخراج نام افراد، مکان‌ها، تاریخ‌ها و …
	2.	پیکره‌ها و منابع واژگانی
	•	بیش از ۵۰ دیتاست و پیکره آماده دارد، از جمله:
	•	WordNet (واژگان انگلیسی)
	•	Brown Corpus (متون کلاسیک انگلیسی)
	•	Movie Reviews (بررسی احساسات فیلم‌ها)
	•	Gutenberg Corpus (متون ادبی کلاسیک)
	3.	ابزارهای زبان‌شناسی
	•	تجزیه و ترسیم درخت‌های نحوی
	•	بررسی فراوانی کلمات و همایندی‌ها
	•	تحلیل هم‌آیی واژه‌ها (مثل: “چای داغ”)
	4.	یادگیری ماشین
	•	پشتیبانی از دسته‌بندی متن
	•	الگوریتم‌هایی مثل Naive Bayes، درخت تصمیم، مدل‌های آنتروپی بیشینه
	•	قابلیت اتصال به scikit-learn برای مدل‌های پیشرفته‌تر
	5.	تمرکز آموزشی
	•	به خاطر سادگی و مستندات زیاد، در دانشگاه‌ها و آموزش NLP بسیار استفاده می‌شود.
---

🛠 نصب

pip install nltk

دانلود منابع:

import nltk
nltk.download('all')

(یا می‌توان فقط بسته‌های موردنیاز را دانلود کرد مثل: nltk.download('punkt'))

---

🔑 ماژول‌های اصلی NLTK
	1.	nltk.tokenize → توکنیزه کردن (تقسیم متن)
	2.	nltk.corpus → دسترسی به دیتاست‌ها
	3.	nltk.stem → ریشه‌یابی و لماتیزه کردن
	4.	nltk.tag → برچسب‌گذاری اجزای کلام
	5.	nltk.chunk → شناسایی موجودیت‌های نامدار
	6.	nltk.parse → تجزیه نحوی
	7.	nltk.classify → دسته‌بندی متن
	8.	nltk.probability → توزیع فراوانی
	9.	nltk.sentiment → تحلیل احساسات

---

📝 نمونه کدها

۱. توکنیزه کردن
```python
from nltk.tokenize import word_tokenize, sent_tokenize

text = "NLTK is a great library for NLP. It makes text analysis easy!"
print(word_tokenize(text))
print(sent_tokenize(text))

۲. ریشه‌یابی و لماتیزه کردن

from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print(stemmer.stem("running"))     # run
print(lemmatizer.lemmatize("better", pos="a"))  # good
```
۳. برچسب‌گذاری اجزای کلام
```python
import nltk
from nltk.tokenize import word_tokenize

nltk.download('averaged_perceptron_tagger')
words = word_tokenize("The quick brown fox jumps over the lazy dog.")
print(nltk.pos_tag(words))
```
۴. شناسایی موجودیت‌ها (NER)

from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

nltk.download('maxent_ne_chunker')
nltk.download('words')

text = "Barack Obama was born in Hawaii."
tokens = word_tokenize(text)
tags = pos_tag(tokens)
tree = ne_chunk(tags)

for subtree in tree:
    if isinstance(subtree, Tree):
        print(subtree.label(), " ".join([token for token, pos in subtree.leaves()]))

۵. توزیع فراوانی

from nltk import FreqDist

words = word_tokenize("NLTK is great. NLTK makes NLP simple. NLP is powerful.")
fdist = FreqDist(words)
print(fdist.most_common(3))


---

✅ مزایای NLTK
	•	یادگیری آسان (مناسب برای مبتدیان و آموزش)
	•	مجموعه بزرگی از دیتاست‌ها و منابع آماده
	•	توابع کاربردی فراوان برای پردازش متن
	•	پشتیبانی قوی در حوزه زبان‌شناسی

⚠️ محدودیت‌ها
	•	سرعت پایین‌تر نسبت به کتابخانه‌های جدیدتر مثل spaCy
	•	مناسب نبودن برای تولید (Production) → بیشتر برای تحقیق و آموزش استفاده می‌شود
	•	مدل‌های قدیمی یادگیری ماشین (بدون پشتیبانی از روش‌های عمیق)

---

🔮 جایگزین‌ها
	•	spaCy → سریع‌تر و مدرن‌تر، مناسب تولید
	•	TextBlob → ساده‌تر برای کارهای مقدماتی
	•	Hugging Face Transformers → پیشرفته‌ترین مدل‌های یادگیری عمیق NLP

---

✅ خلاصه:
NLTK یک کتابخانه پایه و آموزشی برای NLP در پایتون است. برای یادگیری و آزمایش عالی است، اما در پروژه‌های بزرگ و کاربردی معمولاً در کنار spaCy یا Transformers استفاده می‌شود
