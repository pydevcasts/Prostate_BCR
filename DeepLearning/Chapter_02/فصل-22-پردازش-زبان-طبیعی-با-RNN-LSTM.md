# فصل ۲۲ 💬 پردازش زبان طبیعی با RNN/LSTM

## 🎯 اهداف فصل

در فصل‌های قبل ابزارهای پایه‌ای برای کار با متن (پیش‌پردازش، Embedding) و معماری‌های بازگشتی (RNN، LSTM، GRU) را یاد گرفتیم. در این فصل این دو را در دو مسئله‌ی کلاسیک و بسیار پرکاربرد پردازش زبان طبیعی ترکیب می‌کنیم: طبقه‌بندی متن (که شامل تحلیل احساسات هم می‌شود) و ترجمه‌ی ماشینی با معماری Seq2Seq. در پایان این فصل خواهید توانست یک مدل کامل طبقه‌بندی متن چندکلاسه با LSTM بسازید، معماری Seq2Seq (دنباله به دنباله) را برای ترجمه‌ی ماشینی توضیح دهید و پیاده‌سازی کنید، مفهوم Teacher Forcing در آموزش Seq2Seq را درک کنید، و یک مدل ترجمه‌ی ساده را از صفر تا انتها آموزش دهید.

---

# 🏷️ طبقه‌بندی متن: فراتر از تحلیل احساسات دوکلاسه

تا اینجا تحلیل احساسات را عمدتاً به‌صورت دوکلاسه (مثبت/منفی) دیدیم. اما در دنیای واقعی، طبقه‌بندی متن می‌تواند چندکلاسه باشد: دسته‌بندی موضوع اخبار (سیاسی، ورزشی، اقتصادی)، تشخیص قصد کاربر در چت‌بات‌ها، یا تشخیص سطح فوریت تیکت‌های پشتیبانی. معماری پایه برای همه‌ی این مسائل مشابه است؛ تفاوت اصلی در تعداد نورون‌های لایه‌ی خروجی و نوع تابع هزینه است.

---

# 💻 پروژه‌ی اول: طبقه‌بندی موضوع اخبار با LSTM

در این پروژه از دیتاست واقعی AG News استفاده می‌کنیم که شامل عناوین خبری در چهار دسته‌ی جهانی، ورزشی، تجاری و علمی-فناوری است.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ---- بارگذاری دیتاست AG News ----
try:
    from datasets import load_dataset
    dataset = load_dataset("ag_news")
    train_texts = dataset['train']['text'][:5000]
    train_labels = dataset['train']['label'][:5000]
    test_texts = dataset['test']['text'][:1000]
    test_labels = dataset['test']['label'][:1000]
    print("دیتاست AG News با موفقیت بارگذاری شد.")
except ImportError:
    print("کتابخانه datasets نصب نیست. اجرا با داده ساختگی...")
    categories = ['world', 'sports', 'business', 'sci/tech']
    np.random.seed(42)
    train_texts = [f"sample news about {categories[i%4]} topic number {i}"
                   for i in range(500)]
    train_labels = [i % 4 for i in range(500)]
    test_texts = [f"test news about {categories[i%4]} story {i}"
                  for i in range(100)]
    test_labels = [i % 4 for i in range(100)]

class_names = ['جهانی', 'ورزشی', 'تجاری', 'علمی-فناوری']
print(f"تعداد نمونه آموزشی: {len(train_texts)}")
print(f"تعداد کلاس‌ها: {len(class_names)}")

# ---- ساخت واژگان ----
def simple_tokenize(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    return text.split()

word_counts = Counter()
for text in train_texts:
    word_counts.update(simple_tokenize(text))

vocab = ['<PAD>', '<UNK>'] + [
    w for w, c in word_counts.most_common(10000) if c >= 2
]
word_to_idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
print(f"اندازه واژگان: {vocab_size}")

# ---- دیتاست ----
class NewsDataset(Dataset):

    def __init__(self, texts, labels, word_to_idx, max_len=50):
        self.data = []
        for text, label in zip(texts, labels):
            tokens = simple_tokenize(text)[:max_len]
            indices = [word_to_idx.get(t, 1) for t in tokens]
            if len(indices) == 0:
                indices = [1]
            self.data.append((
                torch.tensor(indices, dtype=torch.long),
                torch.tensor(label, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    return texts_padded, torch.stack(labels)


train_dataset = NewsDataset(train_texts, train_labels, word_to_idx)
test_dataset = NewsDataset(test_texts, test_labels, word_to_idx)

trainloader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)
testloader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
)

# ---- مدل طبقه‌بندی چندکلاسه ----
class NewsClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        # Attention ساده برای تمرکز روی کلمات مهم
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)

        # وزن‌دهی Attention به تمام گام‌های زمانی
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (attn_weights * lstm_out).sum(dim=1)

        return self.classifier(context)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NewsClassifier(
    vocab_size=vocab_size, embed_dim=100,
    hidden_size=128, num_classes=len(class_names)
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"تعداد پارامترها: {n_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

for epoch in range(15):

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for texts, labels in trainloader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for texts, labels in testloader:
            texts, labels = texts.to(device), labels.to(device)
            out = model(texts)
            test_correct += (out.argmax(1) == labels).sum().item()
            test_total += labels.size(0)

    train_acc = 100 * correct / total
    test_acc = 100 * test_correct / test_total
    scheduler.step(total_loss)

    if epoch % 3 == 0 or epoch == 14:
        print(f"دوره {epoch+1:2d} | آموزش: {train_acc:.1f}% | آزمون: {test_acc:.1f}%")

print(f"\nدقت نهایی: {test_acc:.2f}%")
```

---

# 🌐 معماری Seq2Seq: پایه‌ی ترجمه‌ی ماشینی

مسئله‌ی ترجمه‌ی ماشینی یک ویژگی خاص دارد که آن را از طبقه‌بندی متمایز می‌کند: هم ورودی و هم خروجی، توالی هستند و طول آن‌ها لزوماً برابر نیست. یک جمله‌ی پنج‌کلمه‌ای انگلیسی ممکن است به یک جمله‌ی هفت‌کلمه‌ای فارسی ترجمه شود. معماری Seq2Seq که در سال ۲۰۱۴ معرفی شد، این مسئله را با ترکیب دو شبکه‌ی بازگشتی حل می‌کند: Encoder که جمله‌ی ورودی را می‌خواند و آن را در یک بردار زمینه خلاصه می‌کند، و Decoder که از این بردار زمینه برای تولید جمله‌ی خروجی کلمه به کلمه استفاده می‌کند.

فرآیند Encoder ساده است: کل توالی ورودی از یک LSTM یا GRU عبور می‌کند و آخرین حالت پنهان آن (که خلاصه‌ای از کل جمله است) به‌عنوان حالت اولیه‌ی Decoder استفاده می‌شود. Decoder سپس به‌صورت خودبازگشتی (Autoregressive) عمل می‌کند: در هر گام یک کلمه تولید می‌کند و همان کلمه را به‌عنوان ورودی گام بعدی استفاده می‌کند، تا زمانی که یک توکن پایان (مثل `<EOS>`) تولید شود.

📷 [تصویر اینجا قرار گیرد: نمودار معماری Seq2Seq با Encoder در سمت چپ که جمله ورودی را پردازش می‌کند و حالت پنهان نهایی را به Decoder در سمت راست منتقل می‌کند که جمله خروجی را کلمه به کلمه تولید می‌کند]

---

# 🎓 Teacher Forcing: ترفند آموزشی مهم

اگر در طول آموزش، Decoder همیشه کلمه‌ای را که خودش در گام قبل تولید کرده به‌عنوان ورودی گام بعد بگیرد، یک مشکل پیش می‌آید: در ابتدای آموزش که مدل هنوز خوب یاد نگرفته، کلمات اشتباه تولید می‌شوند و این خطاها در طول توالی انباشته می‌شوند و آموزش را کند و ناپایدار می‌کنند. تکنیک Teacher Forcing این مشکل را حل می‌کند: در طول آموزش، به‌جای استفاده از خروجی واقعی مدل، کلمه‌ی درست (از داده‌ی هدف) به‌عنوان ورودی گام بعد به Decoder داده می‌شود.

در عمل، معمولاً از یک نسبت Teacher Forcing استفاده می‌شود؛ مثلاً با احتمال ۵۰ درصد از کلمه‌ی واقعی و با احتمال ۵۰ درصد از پیش‌بینی مدل استفاده می‌شود. این تعادل به مدل کمک می‌کند هم سریع‌تر آموزش ببیند و هم برای زمان استنتاج واقعی (که کلمه‌ی واقعی در دسترس نیست) آماده باشد.

---

# 💻 پروژه‌ی دوم: مدل Seq2Seq ساده برای ترجمه

در این پروژه یک مدل Seq2Seq کوچک برای ترجمه‌ی عبارات ساده‌ی انگلیسی به فرانسوی می‌سازیم تا مفاهیم اصلی این معماری را ملموس کنیم.

```python
import random

# ---- داده‌ی نمونه برای ترجمه ----
translation_pairs = [
    ("hello", "bonjour"),
    ("good morning", "bonjour"),
    ("how are you", "comment allez vous"),
    ("thank you", "merci"),
    ("good night", "bonne nuit"),
    ("see you soon", "a bientot"),
    ("i am fine", "je vais bien"),
    ("what is your name", "comment vous appelez vous"),
    ("nice to meet you", "ravi de vous rencontrer"),
    ("goodbye", "au revoir"),
] * 10  # تکرار برای داده کافی

SOS_TOKEN = 0
EOS_TOKEN = 1

def build_vocab(sentences):
    vocab = {'<SOS>': SOS_TOKEN, '<EOS>': EOS_TOKEN}
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

src_sentences = [p[0] for p in translation_pairs]
tgt_sentences = [p[1] for p in translation_pairs]

src_vocab = build_vocab(src_sentences)
tgt_vocab = build_vocab(tgt_sentences)
tgt_idx_to_word = {i: w for w, i in tgt_vocab.items()}

def sentence_to_indices(sentence, vocab):
    indices = [vocab[w] for w in sentence.split()]
    indices.append(EOS_TOKEN)
    return torch.tensor(indices, dtype=torch.long)

print(f"اندازه واژگان مبدا: {len(src_vocab)}")
print(f"اندازه واژگان مقصد: {len(tgt_vocab)}")

# ---- Encoder ----
class Encoder(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


# ---- Decoder ----
class Decoder(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        prediction = self.out(output)
        return prediction, hidden


# ---- مدل کامل Seq2Seq ----
class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.out.out_features

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # اجرای Encoder
        _, hidden = self.encoder(src)

        # اولین ورودی Decoder همیشه <SOS> است
        decoder_input = tgt[:, 0].unsqueeze(1)

        for t in range(1, tgt_len):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t, :] = output.squeeze(1)

            # تصمیم‌گیری برای Teacher Forcing
            use_teacher_forcing = random.random() < teacher_forcing_ratio

            if use_teacher_forcing:
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                decoder_input = output.argmax(2)

        return outputs

    def translate(self, src, max_len=15):
        """ترجمه در زمان استنتاج (بدون Teacher Forcing)"""
        self.eval()
        with torch.no_grad():
            _, hidden = self.encoder(src)
            decoder_input = torch.tensor([[SOS_TOKEN]]).to(self.device)
            result = []

            for _ in range(max_len):
                output, hidden = self.decoder(decoder_input, hidden)
                top_idx = output.argmax(2).item()
                if top_idx == EOS_TOKEN:
                    break
                result.append(top_idx)
                decoder_input = torch.tensor([[top_idx]]).to(self.device)

            return result


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embed_dim = 32
hidden_size = 64

encoder = Encoder(len(src_vocab), embed_dim, hidden_size).to(device)
decoder = Decoder(len(tgt_vocab), embed_dim, hidden_size).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"تعداد پارامترها: {n_params:,}")

optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# ---- آموزش ----
print("\n=== آموزش مدل Seq2Seq ===\n")

for epoch in range(200):

    model.train()
    total_loss = 0.0

    # آموزش با یک جفت در هر گام (برای سادگی)
    indices = list(range(len(translation_pairs)))
    random.shuffle(indices)

    for idx in indices[:20]:  # نمونه‌برداری برای سرعت
        src_text, tgt_text = translation_pairs[idx]
        src_tensor = sentence_to_indices(src_text, src_vocab).unsqueeze(0).to(device)
        tgt_indices = [SOS_TOKEN] + [tgt_vocab[w] for w in tgt_text.split()] + [EOS_TOKEN]
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(0).to(device)

        optimizer.zero_grad()
        outputs = model(src_tensor, tgt_tensor, teacher_forcing_ratio=0.5)

        loss = criterion(
            outputs[:, 1:, :].reshape(-1, len(tgt_vocab)),
            tgt_tensor[:, 1:].reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    if epoch % 40 == 0 or epoch == 199:
        print(f"دوره {epoch+1:3d} | خطا: {total_loss/20:.4f}")

# ---- آزمایش ترجمه ----
print("\n=== نتایج ترجمه ===\n")

test_sentences = ["hello", "thank you", "good night", "how are you"]

for sentence in test_sentences:
    if all(w in src_vocab for w in sentence.split()):
        src_tensor = sentence_to_indices(sentence, src_vocab).unsqueeze(0).to(device)
        translated_indices = model.translate(src_tensor)
        translated_words = [tgt_idx_to_word[i] for i in translated_indices]
        print(f"انگلیسی: '{sentence}'")
        print(f"فرانسوی (پیش‌بینی مدل): '{' '.join(translated_words)}'\n")
```

این پیاده‌سازی هرچند ساده، تمام اجزای اصلی یک سیستم ترجمه‌ی ماشینی واقعی را در خود دارد: Encoder که جمله را به یک نمایش فشرده تبدیل می‌کند، Decoder که این نمایش را گام‌به‌گام به جمله‌ی هدف تبدیل می‌کند، Teacher Forcing برای آموزش پایدارتر، و یک تابع `translate` جداگانه برای زمان استنتاج که از Teacher Forcing استفاده نمی‌کند (چون در دنیای واقعی جمله‌ی هدف را نداریم). در عمل، مدل‌های ترجمه‌ی واقعی از معماری ترنسفورمر (فصل ۲۰) به‌جای RNN استفاده می‌کنند، چون می‌توانند موازی‌سازی شوند و کیفیت بالاتری دارند، اما اصول پایه‌ای Encoder-Decoder که اینجا یاد گرفتیم، در آن‌ها هم وجود دارد.

---

# ❓ سؤالات تستی

### سؤال ۱

در معماری Seq2Seq، وظیفه‌ی اصلی Encoder چیست؟

الف) تولید کلمات خروجی یکی‌یکی

ب) خواندن کل توالی ورودی و فشرده‌کردن آن در یک یا چند بردار حالت پنهان که خلاصه‌ای از معنای کل جمله است

ج) محاسبه‌ی تابع هزینه

د) ترجمه‌ی مستقیم کلمه به کلمه بدون نیاز به Decoder

### سؤال ۲

چرا از Teacher Forcing در آموزش Seq2Seq استفاده می‌شود؟

الف) برای کاهش تعداد پارامترهای مدل

ب) برای جلوگیری از انباشته‌شدن خطاهای پیش‌بینی مدل در طول توالی در مراحل اولیه‌ی آموزش، که با دادن کلمه‌ی درست به‌جای پیش‌بینی نادرست مدل، آموزش را سریع‌تر و پایدارتر می‌کند

ج) چون Decoder بدون Teacher Forcing اصلاً کار نمی‌کند

د) برای افزایش سرعت محاسبات روی GPU

---

## ✅ پاسخ‌ها

**پاسخ سؤال ۱: گزینه ب**

Encoder مثل یک «خواننده» عمل می‌کند که کل جمله‌ی ورودی را می‌خواند و معنای آن را در یک بردار حالت پنهان فشرده می‌کند. این بردار سپس به Decoder داده می‌شود تا بر اساس آن، جمله‌ی خروجی را تولید کند.

**پاسخ سؤال ۲: گزینه ب**

در ابتدای آموزش که مدل ضعیف عمل می‌کند، اگر Decoder از پیش‌بینی‌های اشتباه خودش برای ادامه‌ی تولید استفاده کند، خطاها در طول توالی تجمیع و بزرگ‌تر می‌شوند که آموزش را بسیار کند و ناپایدار می‌کند. Teacher Forcing با دادن کلمه‌ی واقعی هدف به Decoder، این مسیر خطا را قطع می‌کند و امکان یادگیری سریع‌تر را فراهم می‌سازد.

---

# 📝 خلاصه فصل

در این فصل دو کاربرد اصلی RNN/LSTM در پردازش زبان طبیعی را بررسی کردیم. در پروژه‌ی اول، یک مدل طبقه‌بندی متن چندکلاسه با Bidirectional LSTM و Attention ساده روی دیتاست واقعی AG News ساختیم. در پروژه‌ی دوم، با معماری Seq2Seq آشنا شدیم که با ترکیب Encoder و Decoder، توالی‌هایی با طول متفاوت را به هم نگاشت می‌دهد؛ این معماری پایه‌ی سیستم‌های ترجمه‌ی ماشینی است. تکنیک Teacher Forcing را که آموزش این مدل‌ها را پایدارتر می‌کند، درک کردیم. هرچند مدل‌های مدرن از ترنسفورمر برای این مسائل استفاده می‌کنند، درک عمیق Seq2Seq زمینه‌ی محکمی برای فهم معماری‌های پیشرفته‌تر در فصل‌های بعدی فراهم می‌کند.

---

# 🎯 تمرین‌های پایان فصل

۱. در پروژه‌ی اول، Attention را حذف کنید و فقط از آخرین حالت پنهان LSTM استفاده کنید؛ نتایج را مقایسه کنید.

۲. teacher_forcing_ratio را از ۰.۵ به ۱.۰ و سپس ۰.۰ تغییر دهید و تأثیر آن بر کیفیت ترجمه را مشاهده کنید.

۳. به مدل Seq2Seq یک مکانیزم Attention (مشابه فصل ۱۹) اضافه کنید تا Decoder بتواند در هر گام به بخش‌های مختلف ورودی توجه کند.

۴. دیتاست کامل AG News (بدون محدودیت ۵۰۰۰ نمونه) را دانلود و مدل را با آن آموزش دهید.

۵. یک معیار ارزیابی BLEU Score ساده برای سنجش کیفیت ترجمه‌های تولیدشده پیاده‌سازی کنید.

---

در فصل ۲۳ با **مدل‌های زبانی مبتنی بر ترنسفورمر** آشنا می‌شویم؛ BERT و خانواده‌ی GPT را از نظر مفهومی بررسی می‌کنیم و یاد می‌گیریم چگونه با کتابخانه‌ی Hugging Face Transformers از مدل‌های از‌پیش‌آموزش‌دیده‌ی این خانواده در پروژه‌های واقعی استفاده کنیم.
