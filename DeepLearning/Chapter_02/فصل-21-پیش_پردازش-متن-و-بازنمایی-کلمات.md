# فصل ۲۱ 📝 پیش‌پردازش متن و بازنمایی کلمات

## 🎯 اهداف فصل

در فصل‌های قبل برای کار با متن از اندیس‌های عددی ساده استفاده کردیم. اما مدل‌های زبانی واقعی نیاز به پیش‌پردازش دقیق‌تر و بازنمایی‌های بهتری از کلمات دارند. در این فصل یاد می‌گیریم که چگونه متن خام را از ابتدا آماده کنیم و چگونه معنای هر کلمه را به یک بردار عددی تبدیل کنیم که روابط معنایی بین کلمات را حفظ کند. در پایان این فصل خواهید توانست پیش‌پردازش متن شامل Tokenization، حذف کلمات توقف و Stemming/Lemmatization را پیاده‌سازی کنید، مفهوم Word Embedding را توضیح دهید و با بازنمایی‌های ساده مقایسه کنید، روش‌های Word2Vec شامل CBOW و Skip-gram را با مثال عددی محاسبه کنید، و از وردوک‌های از‌پیش‌آموزش‌دیده GloVe در یک مدل واقعی استفاده کنید.

---

# 🔤 چرا به پیش‌پردازش متن نیاز داریم؟

متن خام که از منابع مختلف می‌آید معمولاً پر از نویز، ناسازگاری و اطلاعات اضافه است. جمله‌ی «من دیروز یک کتاب خریدم!!!» و «من دیروز یک کتاب خریدم» از نظر محتوا یکسان هستند اما از نظر ساختار متفاوت. علامت‌های سجاوندی اضافی، حروف بزرگ و کوچک، فاصله‌های اضافی، و کلماتی که تکرار زیادی دارند اما اطلاعات کمی می‌دهند (مثل «و»، «یا»، «را») همگی باید با روش‌های مناسب مدیریت شوند. هدف پیش‌پردازش این است که سیگنال معنایی مفید را از نویز جدا کنیم تا مدل بتواند روابط واقعی را بهتر یاد بگیرد.

---

# 🔪 Tokenization: تقسیم متن به واحدهای معنادار

اولین و مهم‌ترین مرحله‌ی پیش‌پردازش، Tokenization است که متن را به واحدهای کوچک‌تر به نام توکن تقسیم می‌کند. ساده‌ترین نوع آن Word Tokenization است که متن را بر اساس فاصله تقسیم می‌کند، اما در عمل این روش مشکلات زیادی دارد. کلماتی مثل «نمی‌توانم» یا کلمات مرکب، با تقسیم بر اساس فاصله‌ی ساده به‌درستی پردازش نمی‌شوند. Character Tokenization هر کاراکتر را به‌عنوان یک توکن جداگانه در نظر می‌گیرد که واژگان بسیار کوچکی می‌سازد اما توالی‌های بسیار طولانی تولید می‌کند.

روش رایج در مدل‌های مدرن، Subword Tokenization است که کلمات را به زیرکلمات تقسیم می‌کند. در این روش که الگوریتم‌هایی مثل BPE (Byte Pair Encoding) یا WordPiece از آن استفاده می‌کنند، کلمه‌ی رایج «running» ممکن است به «run» و «##ing» تقسیم شود، در حالی که کلمه‌ی نادر «antidisestablishmentarianism» به چندین قطعه‌ی کوچک‌تر تقسیم می‌شود. این رویکرد تعادل خوبی میان اندازه‌ی واژگان و طول توالی برقرار می‌کند و برای کلمات ناشناخته هم کار می‌کند.

---

# 🛑 کلمات توقف (Stop Words)

کلمات توقف کلماتی هستند که به‌قدری رایج‌اند که معمولاً اطلاعات معنایی مفید کمی حمل می‌کنند. در زبان انگلیسی کلماتی مثل «the»، «is»، «at» و «which» و در فارسی کلماتی مثل «و»، «یا»، «که» و «را» جزء کلمات توقف محسوب می‌شوند. حذف آن‌ها حجم داده را کاهش می‌دهد و در بسیاری از مسائل (مثل طبقه‌بندی موضوع) به یادگیری بهتر کمک می‌کند. اما در مسائلی مثل ترجمه‌ی ماشینی یا تولید متن، حذف کلمات توقف می‌تواند مضر باشد چون ساختار دستوری جمله را خراب می‌کند.

---

# 🌿 Stemming و Lemmatization

دو تکنیک دیگر که برای کاهش تنوع واژگان استفاده می‌شوند، Stemming و Lemmatization هستند. Stemming ریشه‌ی کلمه را با حذف پسوند و پیشوند استخراج می‌کند؛ برای مثال «running»، «runner» و «ran» همه به «run» تبدیل می‌شوند. این روش ساده و سریع است اما گاهی ریشه‌های نادرست می‌دهد (مثلاً «caring» به «car» تبدیل می‌شود). Lemmatization که دقیق‌تر است، شکل پایه‌ی کلمه در لغت‌نامه را پیدا می‌کند؛ برای مثال «better» به «good» و «ran» به «run» تبدیل می‌شود اما به اطلاعات دستوری نیاز دارد و کندتر از Stemming است.

---

# 🔢 بازنمایی کلمات: از One-Hot تا Embedding

ساده‌ترین روش برای تبدیل کلمه به عدد، بازنمایی One-Hot است که یک بردار به‌اندازه‌ی واژگان می‌سازد که فقط یک عنصر آن ۱ و بقیه ۰ هستند. اگر واژگان شامل ۵۰،۰۰۰ کلمه باشد، هر کلمه با یک بردار ۵۰،۰۰۰ بعدی نمایش داده می‌شود. این روش دو مشکل اساسی دارد: اول، بردارها بسیار بزرگ و پراکنده (Sparse) هستند که محاسبات را سنگین می‌کند. دوم و مهم‌تر، این بازنمایی هیچ اطلاعاتی درباره‌ی روابط معنایی بین کلمات ندارد؛ فاصله‌ی «پادشاه» از «ملکه» دقیقاً برابر فاصله‌ی «پادشاه» از «ماهی» است.

Word Embedding یا بردار کلمه، راه‌حل این مشکل است. در این روش هر کلمه به یک بردار کوچک‌تر (معمولاً ۵۰ تا ۳۰۰ بعد) تبدیل می‌شود که کلمات با معنای مشابه بردارهای نزدیک به هم دارند. مشهورترین ویژگی این بردارها این است که روابط معنایی در آن‌ها به‌صورت عملیات برداری قابل بیان هستند؛ برای مثال بردار «پادشاه» منهای بردار «مرد» به‌علاوه‌ی بردار «زن» تقریباً برابر بردار «ملکه» می‌شود.

📷 [تصویر اینجا قرار گیرد: نمودار دوبعدی از چندین بردار کلمه که نشان می‌دهد کلمات مشابه از نظر معنا کنار هم قرار گرفته‌اند و رابطه‌ی «پادشاه - مرد + زن = ملکه» به‌صورت بردار نشان داده شده]

---

# 🧠 Word2Vec: یادگیری بردار کلمات از زمینه

Word2Vec که در سال ۲۰۱۳ توسط میکولوف و همکارانش در گوگل معرفی شد، یک روش برای یادگیری خودکار بردار کلمات از روی متن بزرگ است. ایده‌ی اصلی آن از فرضیه‌ی توزیعی در زبان‌شناسی می‌آید: کلماتی که در زمینه‌های مشابه ظاهر می‌شوند، معنای مشابه دارند. Word2Vec دو معماری اصلی دارد:

**روش اول — CBOW (Continuous Bag of Words)**: کلمه‌ی مرکزی را از روی کلمات اطرافش پیش‌بینی می‌کند. اگر جمله «من دیروز به مدرسه رفتم» باشد و پنجره‌ی زمینه‌ای دو کلمه در هر طرف داشته باشیم، مدل باید کلمه‌ی «به» را از روی «دیروز»، «من»، «مدرسه» و «رفتم» پیش‌بینی کند.

**روش دوم — Skip-gram**: کلمات زمینه را از روی کلمه‌ی مرکزی پیش‌بینی می‌کند. با همان مثال، مدل باید با دادن «به»، کلمات «دیروز»، «من»، «مدرسه» و «رفتم» را پیش‌بینی کند. Skip-gram برای کلمات نادر بهتر عمل می‌کند اما کندتر از CBOW آموزش می‌گیرد.

---

# 🔢 مثال عددی ساده‌ی Word2Vec (Skip-gram)

فرض کنید واژگان ما فقط پنج کلمه دارد: {«من»، «دوست»، «دارم»، «یادگیری»، «عمیق»}. کلمه‌ی مرکزی «دوست» (اندیس ۱) است و می‌خواهیم «من» و «دارم» (کلمات کنارش) را پیش‌بینی کنیم. ابعاد Embedding برابر ۲ است.

ماتریس Embedding W_input (5×2):
```
    بُعد ۱  بُعد ۲
من:  [0.1,   0.5]
دوست:[0.3,   0.2]
دارم:[0.6,   0.1]
یادگیری:[0.2, 0.4]
عمیق:[0.4,  0.3]
```

بردار کلمه‌ی مرکزی «دوست»: v = [0.3, 0.2]

ماتریس W_output (5×2) برای تبدیل به احتمال هر کلمه:
```
man:   [0.2, 0.4]   → امتیاز: 0.3×0.2 + 0.2×0.4 = 0.06+0.08 = 0.14
دوست:  [0.5, 0.1]   → امتیاز: 0.3×0.5 + 0.2×0.1 = 0.15+0.02 = 0.17
دارم:  [0.3, 0.6]   → امتیاز: 0.3×0.3 + 0.2×0.6 = 0.09+0.12 = 0.21
یادگیری:[0.1, 0.3]  → امتیاز: 0.3×0.1 + 0.2×0.3 = 0.03+0.06 = 0.09
عمیق: [0.4, 0.2]   → امتیاز: 0.3×0.4 + 0.2×0.2 = 0.12+0.04 = 0.16
```

Softmax این امتیازها یک توزیع احتمال می‌سازد. خطا با مقایسه‌ی این توزیع با کلمه‌ی هدف («من») محاسبه می‌شود و با پس‌انتشار خطا، هر دو ماتریس به‌روز می‌شوند. پس از آموزش روی میلیاردها جمله، ماتریس W_input همان بردار کلمات نهایی Word2Vec است.

---

# 🌐 GloVe: ترکیب آماری و بردار

GloVe (Global Vectors for Word Representation) که در سال ۲۰۱۴ معرفی شد، رویکرد متفاوتی دارد. به‌جای پردازش جمله به جمله مثل Word2Vec، ابتدا یک ماتریس هم‌رخدادی (Co-occurrence Matrix) می‌سازد که نشان می‌دهد هر جفت کلمه چند بار در کنار هم آمده‌اند، سپس بردار کلمات را به‌گونه‌ای یاد می‌گیرد که حاصل‌ضرب بردارهای دو کلمه با لگاریتم تعداد هم‌رخدادی آن‌ها تناسب داشته باشد. فرمول هدف GloVe به شکل زیر است:

```
J = Σ_{i,j} f(X_{ij}) × (w_i^T × w̃_j + b_i + b̃_j - log(X_{ij}))²
```

در این فرمول، X_{ij} تعداد هم‌رخدادی کلمه‌ی i و j است، w_i و w̃_j بردار کلمات هستند، و f یک تابع وزن‌دهی است که رویدادهای بسیار رایج را کمتر جریمه می‌کند. مزیت اصلی GloVe این است که از آمار کلی متن استفاده می‌کند و می‌تواند الگوهای سراسری را که Word2Vec ممکن است از دست بدهد، یاد بگیرد.

---

# 💻 پروژه‌ی عملی فصل: خط لوله‌ی کامل پیش‌پردازش و استفاده از GloVe

در این پروژه یک خط لوله‌ی کامل پیش‌پردازش متن می‌سازیم و سپس از بردارهای GloVe از‌پیش‌آموزش‌دیده برای بهتر کردن یک مدل طبقه‌بندی متن استفاده می‌کنیم.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import os

# ---- بخش اول: خط لوله‌ی پیش‌پردازش متن ----

class TextPreprocessor:
    """
    یک خط لوله‌ی کامل برای پیش‌پردازش متن انگلیسی
    """

    # کلمات توقف رایج انگلیسی (نسخه‌ی ساده‌شده)
    STOP_WORDS = {
        'a', 'an', 'the', 'is', 'it', 'in', 'on', 'at', 'to',
        'for', 'of', 'and', 'or', 'but', 'not', 'with', 'this',
        'that', 'are', 'was', 'were', 'be', 'been', 'has', 'had',
        'have', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
    }

    def __init__(self, remove_stopwords=True, use_stemming=True):
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming

    def clean_text(self, text):
        """پاک‌سازی متن از HTML، اعداد و کاراکترهای خاص"""
        # حذف تگ‌های HTML
        text = re.sub(r'<[^>]+>', ' ', text)
        # تبدیل به حروف کوچک
        text = text.lower()
        # حذف کاراکترهای خاص و نگه‌داشتن حروف و اعداد
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # حذف فاصله‌های اضافی
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def simple_stem(self, word):
        """
        Stemming ساده: حذف پسوندهای رایج
        نکته: در پروژه‌های واقعی از nltk یا spaCy استفاده کنید
        """
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'ness', 'ment']
        for suffix in sorted(suffixes, key=len, reverse=True):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[:-len(suffix)]
        return word

    def tokenize(self, text):
        """تقسیم متن به توکن‌ها با اعمال پیش‌پردازش"""
        text = self.clean_text(text)
        tokens = text.split()

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.STOP_WORDS]

        if self.use_stemming:
            tokens = [self.simple_stem(t) for t in tokens]

        return tokens

    def build_vocabulary(self, texts, max_vocab=10000, min_freq=2):
        """ساخت واژگان از مجموعه‌ای از متون"""
        counter = Counter()
        for text in texts:
            counter.update(self.tokenize(text))

        # نگه‌داشتن فقط کلماتی که بیش از min_freq بار آمده‌اند
        vocab = ['<PAD>', '<UNK>'] + [
            word for word, freq in counter.most_common(max_vocab)
            if freq >= min_freq
        ]
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab)
        return self.word_to_idx

    def encode(self, text, max_len=100):
        """تبدیل متن به دنباله‌ای از اندیس‌ها"""
        tokens = self.tokenize(text)[:max_len]
        indices = [self.word_to_idx.get(t, 1) for t in tokens]
        # Padding تا طول max_len
        indices += [0] * (max_len - len(indices))
        return indices


# ---- نمایش کارکرد پیش‌پردازش ----
preprocessor = TextPreprocessor(remove_stopwords=True, use_stemming=True)

sample_texts = [
    "This movie was absolutely amazing and wonderful!",
    "The acting was terrible and the plot made no sense.",
    "I really enjoyed watching this film with my friends.",
    "Boring and predictable, not worth your time at all."
]

print("=== نمایش خط لوله‌ی پیش‌پردازش ===\n")
for text in sample_texts:
    tokens = preprocessor.tokenize(text)
    print(f"اصلی: {text}")
    print(f"توکن‌ها: {tokens}\n")

# ساخت واژگان
preprocessor.build_vocabulary(sample_texts, min_freq=1)
print(f"اندازه واژگان: {preprocessor.vocab_size}")
print(f"اندیس 'amaz': {preprocessor.word_to_idx.get('amaz', 'ناشناخته')}")


# ---- بخش دوم: بارگذاری و استفاده از GloVe ----

def load_glove_embeddings(glove_path, word_to_idx, embed_dim=100):
    """
    بارگذاری بردارهای GloVe از‌پیش‌آموزش‌دیده.
    فایل GloVe از: https://nlp.stanford.edu/projects/glove/
    نسخه glove.6B.100d.txt را دانلود کنید.
    """

    vocab_size = len(word_to_idx)
    # مقداردهی اولیه با اعداد تصادفی کوچک
    embeddings = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim))
    # توکن padding با بردار صفر
    embeddings[0] = np.zeros(embed_dim)

    found = 0
    if os.path.exists(glove_path):
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                if word in word_to_idx:
                    idx = word_to_idx[word]
                    embeddings[idx] = np.array(parts[1:], dtype=np.float32)
                    found += 1
        print(f"تعداد کلمات پیداشده در GloVe: {found}/{vocab_size}")
    else:
        print(f"فایل GloVe یافت نشد: {glove_path}")
        print("استفاده از بردارهای تصادفی (برای نمایش ساختار کد)")

    return torch.tensor(embeddings, dtype=torch.float32)


# ---- بخش سوم: مدل با Embedding از GloVe ----

class SentimentModelWithGloVe(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_size,
                 num_classes, pretrained_embeddings=None,
                 freeze_embeddings=False):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # اگر بردارهای از‌پیش‌آموزش‌دیده داریم، آن‌ها را بارگذاری کنیم
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            print(f"بردارهای GloVe با موفقیت بارگذاری شدند.")

        # اگر می‌خواهیم Embedding ثابت باشد (Feature Extraction)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
            print("Embedding ثابت است و در آموزش تغییر نمی‌کند.")

        # یک لایه Bidirectional LSTM
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3
        )

        # طبقه‌بند نهایی
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (h_n, _) = self.lstm(embedded)
        # ترکیب آخرین حالت پنهان از دو جهت
        forward_h = h_n[-2, :, :]
        backward_h = h_n[-1, :, :]
        combined = torch.cat([forward_h, backward_h], dim=1)
        return self.classifier(combined)


# ---- بخش چهارم: آزمایش و مقایسه ----

def run_experiment(use_pretrained, freeze_emb, embed_dim=50, epochs=10):
    """
    آزمایش مدل با و بدون GloVe
    """

    max_seq_len = 50
    hidden_size = 64

    # داده‌ی ساختگی با واژگان ساده
    vocab_size = preprocessor.vocab_size

    # بردارهای GloVe (یا تصادفی اگر فایل نداشتیم)
    glove_path = f'glove.6B.{embed_dim}d.txt'
    pretrained_embs = load_glove_embeddings(
        glove_path, preprocessor.word_to_idx, embed_dim
    ) if use_pretrained else None

    model = SentimentModelWithGloVe(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_classes=2,
        pretrained_embeddings=pretrained_embs,
        freeze_embeddings=freeze_emb
    )

    # داده‌ی ساختگی برای نمایش
    n_samples = 200
    X = torch.randint(0, vocab_size, (n_samples, max_seq_len))
    y = torch.randint(0, 2, (n_samples,))
    X_train, y_train = X[:160], y[:160]
    X_test, y_test = X[160:], y[160:]

    trainloader = DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=32, shuffle=True
    )

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x_b, y_b in trainloader:
            optimizer.zero_grad()
            loss = criterion(model(x_b), y_b)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(1)
        acc = (preds == y_test).float().mean().item() * 100

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return acc, n_trainable


print("\n=== مقایسه‌ی سه حالت مختلف Embedding ===\n")

configs = [
    (False, False, "Embedding تصادفی (آموزش از صفر)"),
    (True, True,  "GloVe ثابت (فقط Feature Extraction)"),
    (True, False, "GloVe + Fine-tuning (بهترین حالت)"),
]

for use_pre, freeze, name in configs:
    acc, params = run_experiment(use_pre, freeze)
    print(f"{name}:")
    print(f"  پارامترهای قابل‌یادگیری: {params:,}")
    print(f"  دقت روی آزمون: {acc:.1f}%\n")

print("نکته: با داده‌ی واقعی IMDB، GloVe+Fine-tuning معمولاً بهترین نتیجه را می‌دهد.")
```

---

# ❓ سؤالات تستی

### سؤال ۱

تفاوت اصلی روش Skip-gram از CBOW در Word2Vec چیست؟

الف) Skip-gram سریع‌تر از CBOW آموزش می‌گیرد

ب) CBOW کلمه‌ی مرکزی را از کلمات اطراف پیش‌بینی می‌کند، در حالی که Skip-gram کلمات اطراف را از کلمه‌ی مرکزی پیش‌بینی می‌کند و برای کلمات نادر بهتر عمل می‌کند

ج) Skip-gram فقط برای زبان‌های اروپایی کار می‌کند

د) CBOW از ماتریس هم‌رخدادی استفاده می‌کند

### سؤال ۲

چرا در پروژه‌ی این فصل، وقتی از GloVe استفاده می‌کنیم، گاهی Embedding را Freeze می‌کنیم؟

الف) چون فایل GloVe Read-Only است

ب) چون وقتی داده‌ی آموزشی کم است، تغییر دادن Embedding‌های از‌پیش‌آموزش‌دیده ممکن است دانش مفید قبلی را از بین ببرد و Freeze کردن از بیش‌برازش جلوگیری می‌کند

ج) چون Freeze کردن همیشه نتیجه‌ی بهتری می‌دهد

د) چون PyTorch از آموزش Embedding از‌پیش‌آموزش‌دیده پشتیبانی نمی‌کند

---

## ✅ پاسخ‌ها

**پاسخ سؤال ۱: گزینه ب**

CBOW با دیدن چندین کلمه‌ی زمینه به‌طور همزمان، کلمه‌ی مرکزی را پیش‌بینی می‌کند که آموزش سریع‌تری دارد. Skip-gram برعکس عمل می‌کند و از یک کلمه می‌خواهد تمام کلمات زمینه را پیش‌بینی کند. چون Skip-gram برای هر کلمه‌ی مرکزی چندین مثال آموزشی می‌سازد، برای کلمات نادر که کم‌تر در داده ظاهر می‌شوند بردارهای بهتری یاد می‌گیرد.

**پاسخ سؤال ۲: گزینه ب**

بردارهای GloVe از آموزش روی میلیاردها کلمه به دست آمده‌اند. وقتی داده‌ی آموزشی ما محدود است، اگر اجازه دهیم Embedding آزادانه تغییر کند، ممکن است مدل این بردارهای ارزشمند را برای تطابق بیشتر با داده‌ی کم دستکاری کند و در نتیجه دانش عمومی زبان را از دست بدهد. Freeze کردن Embedding در این حالت مثل استفاده از Feature Extraction به‌جای Fine-tuning است.

---

# 📝 خلاصه فصل

در این فصل پایه‌ی کار با متن در یادگیری عمیق را از اساس بررسی کردیم. با مراحل اصلی پیش‌پردازش شامل پاک‌سازی، Tokenization، حذف کلمات توقف و Stemming آشنا شدیم. محدودیت بازنمایی One-Hot را فهمیدیم و دیدیم که Word Embedding چطور این مشکل را حل می‌کند. دو روش اصلی Word2Vec یعنی CBOW و Skip-gram را با فرمول و مثال عددی بررسی کردیم. رویکرد GloVe را شناختیم که از ماتریس هم‌رخدادی سراسری استفاده می‌کند. در پروژه‌ی عملی یک خط لوله‌ی کامل پیش‌پردازش ساختیم و سه حالت Embedding (تصادفی، GloVe ثابت، و GloVe + Fine-tuning) را مقایسه کردیم.

---

# 🎯 تمرین‌های پایان فصل

۱. خط لوله‌ی پیش‌پردازش را با کتابخانه‌ی NLTK پیاده‌سازی کنید و Stemming پیشرفته‌تر را امتحان کنید.

۲. بردارهای GloVe را دانلود کنید و کلمات مشابه «king» را با محاسبه‌ی شباهت کسینوسی پیدا کنید.

۳. رابطه‌ی معروف «king - man + woman ≈ queen» را با بردارهای GloVe تأیید کنید.

۴. مدل Skip-gram ساده را روی یک متن کوچک از صفر پیاده‌سازی کنید.

۵. دیتاست IMDB را با سه حالت مختلف Embedding آموزش دهید و نتایج را مقایسه کنید.

---

در فصل ۲۲ وارد دنیای **پردازش زبان طبیعی با شبکه‌های بازگشتی** می‌شویم و مسائلی مثل تحلیل احساسات، طبقه‌بندی متن و مدل‌های Seq2Seq برای ترجمه‌ی ماشینی را با کد کامل و داده‌ی واقعی پیاده‌سازی می‌کنیم.
