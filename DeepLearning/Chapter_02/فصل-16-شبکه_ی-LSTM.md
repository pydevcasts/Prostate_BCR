# فصل ۱۶ 🚪 شبکه‌ی LSTM: یادگیری با حافظه‌ی بلندمدت

## 🎯 اهداف فصل

در دو فصل گذشته مشکل اصلی RNN ساده را هم از نظر نظری و هم از نظر عملی مشاهده کردیم: وقتی توالی‌ها طولانی می‌شوند، گرادیان محو می‌شود و شبکه نمی‌تواند وابستگی‌های دور را یاد بگیرد. در این فصل با راه‌حل اصلی این مشکل آشنا می‌شویم: شبکه‌ی LSTM که مخفف Long Short-Term Memory یا حافظه‌ی کوتاه‌مدت بلند است. این معماری که در سال ۱۹۹۷ توسط زپف هوکرایتر و یورگن اشمیدهوبر معرفی شد، با طراحی هوشمندانه‌ی یک سیستم دروازه‌ای، جریان اطلاعات در طول زمان را کنترل می‌کند. در پایان این فصل خواهید توانست فرمول و نقش هر یک از چهار دروازه‌ی LSTM را با مثال عددی محاسبه کنید، تفاوت Cell State از Hidden State را توضیح دهید، دلیل مقاومت LSTM در برابر گرادیان محوشونده را بیان کنید، و یک LSTM کامل را روی دیتاست واقعی در PyTorch پیاده‌سازی کنید.

---

# 💡 ایده‌ی اصلی LSTM: کنترل جریان اطلاعات

مشکل اساسی RNN ساده این بود که حالت پنهان h_t باید دو کار همزمان انجام دهد: هم حافظه‌ی بلندمدت توالی را نگه دارد و هم اطلاعات مرتبط برای پیش‌بینی فعلی را فراهم کند. این دو وظیفه در تضاد با هم هستند؛ وقتی شبکه اطلاعات جدید را وارد h_t می‌کند، اطلاعات قدیمی از بین می‌روند.

LSTM این مشکل را با یک ایده‌ی ظریف حل می‌کند: یک بزرگراه اطلاعاتی جداگانه به نام حالت سلولی (Cell State) اضافه می‌کند که برای نگهداری حافظه‌ی بلندمدت طراحی شده است. این Cell State مثل یک نوار نقاله است که اطلاعات را در طول توالی حمل می‌کند، و LSTM با استفاده از سه دروازه‌ی هوشمند (Forget، Input و Output) کنترل می‌کند که چه اطلاعاتی به این نوار اضافه شود، چه اطلاعاتی از آن حذف شود، و چه بخشی از آن به خروجی تبدیل شود.

📷 [تصویر اینجا قرار گیرد: نمودار کامل یک سلول LSTM با نمایش دو مسیر موازی: Cell State (خط صاف در بالا) و Hidden State (مسیر پایین با دروازه‌ها)، با رنگ‌بندی متفاوت برای هر دروازه]

---

# 🚪 دروازه‌ی اول: Forget Gate (دروازه‌ی فراموشی)

اولین دروازه تعیین می‌کند چه مقدار از Cell State قبلی (c_{t-1}) باید حفظ شود و چه مقدار باید «فراموش» شود. ورودی این دروازه ترکیبی از حالت پنهان گام قبلی (h_{t-1}) و ورودی فعلی (x_t) است، و خروجی آن یک بردار از مقادیر بین صفر و یک است که با Sigmoid محاسبه می‌شود:

```
f_t = σ( W_f × [h_{t-1}, x_t] + b_f )
```

مقدار نزدیک به یک در این بردار یعنی «این بخش از حافظه را نگه دار» و مقدار نزدیک به صفر یعنی «این بخش را فراموش کن». به‌عنوان یک مثال شهودی، تصور کنید LSTM در حال خواندن یک متن درباره‌ی یک شهر است. وقتی از یک شهر به شهر دیگری می‌رود، Forget Gate اطلاعات مرتبط با شهر قبلی (مثل جنسیت فاعل) را فراموش می‌کند تا برای اطلاعات جدید جا باز شود.

---

# 🚪 دروازه‌ی دوم: Input Gate (دروازه‌ی ورودی)

دومین دروازه تعیین می‌کند چه اطلاعات جدیدی باید به Cell State اضافه شود. این دروازه از دو بخش تشکیل شده است. بخش اول (i_t) با Sigmoid تعیین می‌کند کدام مقادیر باید به‌روز شوند. بخش دوم (g_t یا c̃_t) با Tanh یک بردار از مقادیر احتمالی جدید می‌سازد که می‌توانند به Cell State اضافه شوند:

```
i_t = σ( W_i × [h_{t-1}, x_t] + b_i )
g_t = tanh( W_g × [h_{t-1}, x_t] + b_g )
```

Cell State جدید از ترکیب این دو دروازه و Cell State قبلی به‌روز می‌شود:

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
```

در این فرمول، علامت ⊙ ضرب عنصر‌به‌عنصر (Hadamard Product) را نشان می‌دهد. بخش اول (f_t ⊙ c_{t-1}) همان حافظه‌ی قبلی پس از فراموشی است، و بخش دوم (i_t ⊙ g_t) اطلاعات جدیدی است که اضافه می‌شود.

---

# 🚪 دروازه‌ی سوم: Output Gate (دروازه‌ی خروجی)

سومین دروازه تعیین می‌کند کدام بخش از Cell State به‌عنوان حالت پنهان جدید (h_t) به خروجی ارسال شود:

```
o_t = σ( W_o × [h_{t-1}, x_t] + b_o )
h_t = o_t ⊙ tanh( c_t )
```

ابتدا c_t از Tanh عبور می‌کند تا مقادیر آن به بازه‌ی بین منفی یک و مثبت یک نرمال شود، سپس با o_t ضرب می‌شود تا فقط اطلاعات مرتبط فعلاً به خروجی ارسال شوند. حالت پنهان h_t هم برای تولید خروجی این گام و هم به‌عنوان ورودی گام بعدی استفاده می‌شود.

📷 [تصویر اینجا قرار گیرد: نمودار سه دروازه‌ی LSTM به‌صورت جداگانه با رنگ‌های متفاوت، هر کدام با نمایش فرمول و جهت جریان داده]

---

# 🔢 مثال عددی کامل: یک گام کامل LSTM

برای درک عملی این فرمول‌ها، یک گام کامل از LSTM را با اعداد ساده محاسبه می‌کنیم. فرض کنید اندازه‌ی ورودی یک و اندازه‌ی حالت پنهان دو است. مقادیر اولیه همه صفر هستند و ورودی فعلی برابر ۰.۵ است. برای سادگی، همه‌ی ماتریس‌های وزن را اسکالر فرض می‌کنیم:

ورودی x_t = 0.5، حالت پنهان قبلی h_{t-1} = 0، حالت سلولی قبلی c_{t-1} = 0

با وزن‌های ساده‌شده W_f = W_i = W_g = W_o = 1 و بایاس‌های b_f = -1، b_i = 0، b_g = 0، b_o = 0:

محاسبه‌ی ورودی هر دروازه (z = W × [h, x] + b):
```
z_f = 1×0 + 1×0.5 + (-1) = -0.5
z_i = 1×0 + 1×0.5 + 0   =  0.5
z_g = 1×0 + 1×0.5 + 0   =  0.5
z_o = 1×0 + 1×0.5 + 0   =  0.5
```

اعمال توابع فعال‌سازی:
```
f_t = σ(-0.5) = 1/(1+e^0.5) ≈ 0.378     (فراموشی متوسط)
i_t = σ(0.5)  = 1/(1+e^-0.5) ≈ 0.622    (ورودی متوسط)
g_t = tanh(0.5) ≈ 0.462                  (مقدار پیشنهادی)
o_t = σ(0.5)  ≈ 0.622                    (خروجی متوسط)
```

به‌روزرسانی Cell State:
```
c_t = f_t × c_{t-1} + i_t × g_t
c_t = 0.378 × 0 + 0.622 × 0.462
c_t ≈ 0 + 0.287 = 0.287
```

محاسبه‌ی Hidden State جدید:
```
h_t = o_t × tanh(c_t)
h_t = 0.622 × tanh(0.287)
h_t = 0.622 × 0.280
h_t ≈ 0.174
```

در این گام، ورودی ۰.۵ باعث شد که حافظه‌ی سلولی از صفر به ۰.۲۸۷ برسد (اطلاعات وارد شدند) و حالت پنهان هم به ۰.۱۷۴ رسید. در گام بعدی، این مقادیر به‌عنوان c_{t-1} و h_{t-1} وارد فرمول می‌شوند.

---

# 🛡️ چرا LSTM در برابر گرادیان محوشونده مقاوم است؟

کلید درک مقاومت LSTM در برابر گرادیان محوشونده، در فرمول به‌روزرسانی Cell State نهفته است:

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
```

وقتی گرادیان را از c_t به c_{t-1} منتقل می‌کنیم، مشتق c_t نسبت به c_{t-1} برابر f_t است. این تفاوت اساسی با RNN ساده دارد: در RNN ساده، مشتق h_t نسبت به h_{t-1} شامل ضرب ماتریس W_hh و مشتق Tanh بود که هر دو می‌توانستند باعث کوچک‌شدن گرادیان شوند. اما در LSTM، گرادیان از طریق c_t منتقل می‌شود و ضریب آن f_t است که یک عدد بین صفر و یک است. مهم‌تر از همه، f_t در طول آموزش یاد می‌گیرد که چه وقت باید نزدیک یک باشد (یعنی حافظه را حفظ کن و گرادیان را عبور بده) و چه وقت باید نزدیک صفر باشد (یعنی فراموش کن). این یادگیری تطبیقی جریان گرادیان، راز اصلی موفقیت LSTM است.

---

# 💻 پروژه‌ی عملی فصل: تحلیل احساسات متن با LSTM

در این پروژه از LSTM برای تحلیل احساسات نظرات مشتریان روی دیتاست IMDB (نظرات فیلم) استفاده می‌کنیم. این مسئله نمونه‌ی کلاسیکی است که نیاز به درک وابستگی‌های طولانی در متن دارد، چراکه گاهی کلید احساس یک نظر در اول جمله است و پیش‌بینی باید بعد از خواندن تمام جمله انجام شود.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import numpy as np

# ---- ساخت دیتاست ساده برای نمایش مفاهیم ----
# (برای دیتاست واقعی IMDB از torchtext استفاده کنید)

sample_reviews = [
    ("this movie was absolutely fantastic and wonderful", 1),
    ("terrible film complete waste of time", 0),
    ("amazing acting and great storyline highly recommend", 1),
    ("boring and predictable nothing interesting happened", 0),
    ("one of the best movies i have ever seen", 1),
    ("awful direction bad script poor performances", 0),
    ("loved every minute of this beautiful film", 1),
    ("disappointing and dull definitely not worth watching", 0),
    ("brilliant masterpiece that moved me deeply", 1),
    ("horrible movie with no plot or character development", 0),
]

# ---- ساخت واژگان ----
all_words = []
for text, _ in sample_reviews:
    all_words.extend(text.split())

word_counts = Counter(all_words)
vocab = ['<PAD>', '<UNK>'] + [w for w, c in word_counts.most_common()]
word_to_idx = {w: i for i, w in enumerate(vocab)}

def tokenize(text, word_to_idx, max_len=20):
    tokens = text.split()[:max_len]
    indices = [word_to_idx.get(t, 1) for t in tokens]
    return torch.tensor(indices, dtype=torch.long)

# ---- تعریف دیتاست ----
class SentimentDataset(Dataset):

    def __init__(self, reviews, word_to_idx):
        self.data = [
            (tokenize(text, word_to_idx), torch.tensor(label, dtype=torch.float))
            for text, label in reviews
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    texts, labels = zip(*batch)
    # Padding برای یکسان‌سازی طول توالی‌ها
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    return texts_padded, torch.stack(labels)

# تقسیم داده
train_data = sample_reviews[:8]
test_data = sample_reviews[8:]

train_dataset = SentimentDataset(train_data, word_to_idx)
test_dataset = SentimentDataset(test_data, word_to_idx)

trainloader = DataLoader(
    train_dataset, batch_size=4,
    shuffle=True, collate_fn=collate_fn
)
testloader = DataLoader(
    test_dataset, batch_size=4,
    shuffle=False, collate_fn=collate_fn
)

# ---- تعریف مدل LSTM ----
class SentimentLSTM(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_size,
                 num_layers, dropout=0.3):
        super().__init__()

        # لایه‌ی Embedding: تبدیل اندیس کلمه به بردار
        # هر کلمه به یک بردار embed_dim بعدی تبدیل می‌شود
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0
        )

        # لایه‌ی LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True   # LSTM دوطرفه برای درک بهتر زمینه
        )

        # طبقه‌بند نهایی
        # چون bidirectional=True است، خروجی 2×hidden_size است
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_len) → embedded: (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # اجرای LSTM روی کل توالی
        # lstm_out: (batch, seq_len, hidden_size*2)
        lstm_out, (h_n, c_n) = self.lstm(embedded)

        # استفاده از آخرین حالت پنهان هر دو جهت
        # h_n شکل (num_layers*2, batch, hidden_size) دارد
        # آخرین لایه از هر دو جهت را می‌گیریم و کنار هم می‌گذاریم
        forward_hidden = h_n[-2, :, :]   # آخرین لایه‌ی جهت رو به جلو
        backward_hidden = h_n[-1, :, :]  # آخرین لایه‌ی جهت معکوس
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)

        return self.classifier(combined).squeeze(1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SentimentLSTM(
    vocab_size=len(vocab),
    embed_dim=32,
    hidden_size=64,
    num_layers=2,
    dropout=0.3
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"تعداد پارامترها: {total_params:,}")

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---- آموزش مدل ----
for epoch in range(50):

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for texts, labels in trainloader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(texts)
        loss = criterion(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        predicted = (preds > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    if epoch % 10 == 0 or epoch == 49:
        train_acc = 100 * correct / total
        print(f"دوره {epoch+1:2d} | خطا: {total_loss/len(trainloader):.4f} | "
              f"دقت آموزش: {train_acc:.1f}%")

# ---- ارزیابی نهایی ----
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for texts, labels in testloader:
        texts = texts.to(device)
        preds = model(texts)
        predicted = (preds.cpu() > 0.5).float()
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

test_acc = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"\nدقت نهایی روی داده آزمون: {test_acc:.1f}%")

# ---- آزمایش مدل روی جملات جدید ----
def predict_sentiment(text, model, word_to_idx, device):
    model.eval()
    tokens = tokenize(text, word_to_idx).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = model(tokens).item()
    sentiment = "مثبت 😊" if prob > 0.5 else "منفی 😞"
    print(f"متن: '{text}'")
    print(f"احتمال مثبت: {prob:.3f} — احساس: {sentiment}\n")

print("\nآزمایش روی جملات جدید:")
predict_sentiment("great movie loved it very much", model, word_to_idx, device)
predict_sentiment("bad film terrible waste of money", model, word_to_idx, device)
predict_sentiment("interesting story but somewhat slow pacing", model, word_to_idx, device)
```

در این پیاده‌سازی چند نکته‌ی مهم وجود دارد که باید به آن‌ها توجه کنید. نخست، از `bidirectional=True` استفاده شده است؛ این به مدل اجازه می‌دهد توالی را هم از چپ به راست و هم از راست به چپ پردازش کند، که درک بهتری از زمینه‌ی هر کلمه فراهم می‌کند. دوم، `pad_sequence` برای یکسان‌سازی طول توالی‌های مختلف در یک دسته استفاده شده و `padding_idx=0` در Embedding تضمین می‌کند که پیکسل‌های padding تأثیری بر آموزش ندارند. سوم، آخرین حالت پنهان از هر دو جهت LSTM برای طبقه‌بندی نهایی کنار هم قرار گرفته‌اند. برای استفاده‌ی واقعی، می‌توانید این کد را با دیتاست کامل IMDB از `torchtext` یا `datasets` از Hugging Face اجرا کنید.

---

# ❓ سؤالات تستی

### سؤال ۱

تفاوت اصلی Cell State (c_t) از Hidden State (h_t) در LSTM چیست؟

الف) Cell State فقط برای خروجی استفاده می‌شود و Hidden State برای حافظه

ب) Cell State بزرگراه اصلی انتقال اطلاعات بلندمدت است که با تغییر کمتری از گام‌های زمانی عبور می‌کند، در حالی که Hidden State اطلاعات کوتاه‌مدت‌تری برای پیش‌بینی فعلی فراهم می‌کند

ج) هر دو دقیقاً یکسان هستند

د) Cell State فقط در گام اول استفاده می‌شود

### سؤال ۲

در فرمول c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t، اگر مقدار f_t نزدیک به صفر باشد، چه اتفاقی می‌افتد؟

الف) اطلاعات جدید وارد Cell State نمی‌شود

ب) حافظه‌ی قبلی (c_{t-1}) تقریباً کاملاً فراموش می‌شود و Cell State با اطلاعات جدید بازنویسی می‌شود

ج) خروجی مدل صفر می‌شود

د) گرادیان انفجار پیدا می‌کند

---

## ✅ پاسخ‌ها

**پاسخ سؤال ۱: گزینه ب**

این تفکیک وظایف، یکی از مهم‌ترین نوآوری‌های LSTM است. Cell State مثل یک نوار نقاله‌ی بلند عمل می‌کند که اطلاعات مهم را بدون تغییر زیاد از طول توالی عبور می‌دهد، در حالی که Hidden State اطلاعاتی است که مدل فکر می‌کند برای گام فعلی مفید هستند. این تقسیم‌کار است که LSTM را قادر به یادگیری وابستگی‌های بلندمدت می‌کند.

**پاسخ سؤال ۲: گزینه ب**

Forget Gate با مقدار f_t ≈ 0 ضرب می‌کند که باعث می‌شود تقریباً همه‌ی Cell State قبلی به صفر برسد. این یعنی «همه چیز را فراموش کن و با اطلاعات جدید شروع کن». این مکانیزم برای موقعیت‌هایی مفید است که یک موضوع کاملاً جدید در متن شروع می‌شود و دیگر نیازی به اطلاعات قبلی نیست.

---

# 📝 خلاصه فصل

در این فصل به‌طور کامل با معماری LSTM آشنا شدیم. دیدیم که LSTM با جداسازی وظایف حافظه‌ی بلندمدت (Cell State) و اطلاعات کوتاه‌مدت (Hidden State)، و با استفاده از سه دروازه‌ی هوشمند، مشکل گرادیان محوشونده را حل می‌کند. هر دروازه را با فرمول کامل و یک مثال عددی گام‌به‌گام بررسی کردیم. دلیل ریاضی مقاومت LSTM (انتقال گرادیان از طریق f_t به‌جای ماتریس وزن ثابت) را درک کردیم. در پروژه‌ی عملی یک مدل تحلیل احساسات با Bidirectional LSTM روی داده‌ی متنی پیاده‌سازی کردیم.

---

# 🎯 تمرین‌های پایان فصل

۱. یک گام کامل LSTM را با مقادیر دلخواه متفاوت از مثال این فصل روی کاغذ محاسبه کنید.

۲. LSTM یک‌طرفه را با Bidirectional LSTM مقایسه کنید و ببینید تأثیر آن روی دقت تحلیل احساسات چقدر است.

۳. تعداد لایه‌های LSTM را از ۲ به ۳ افزایش دهید و تأثیر آن بر دقت و زمان آموزش را بررسی کنید.

۴. به‌جای استفاده از حالت پنهان آخرین گام، میانگین تمام حالت‌های پنهان (Mean Pooling) را برای طبقه‌بندی به کار ببرید.

۵. دیتاست کامل IMDB را از کتابخانه‌ی `datasets` بارگذاری کنید و مدل این فصل را روی آن آموزش دهید.

---

در فصل ۱۷ با **معماری GRU** آشنا می‌شویم؛ نسخه‌ی ساده‌شده‌تر و سریع‌تر LSTM که با دو دروازه به‌جای سه دروازه، در بسیاری از مسائل عملکردی مشابه LSTM دارد و درک آن پایه‌ای برای انتخاب هوشمند میان این دو معماری می‌شود.
