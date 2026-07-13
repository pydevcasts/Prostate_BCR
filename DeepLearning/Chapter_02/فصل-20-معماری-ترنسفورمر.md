# فصل ۲۰ 🧩 معماری ترنسفورمر (Transformer)

## 🎯 اهداف فصل

در فصل قبل با مکانیزم توجه آشنا شدیم و دیدیم که Self-Attention چگونه روابط بین تمام موقعیت‌های یک توالی را به‌صورت مستقیم و موازی محاسبه می‌کند. در سال ۲۰۱۷، وسوانی و همکارانش در گوگل مقاله‌ی «Attention is All You Need» را منتشر کردند که با ترکیب چندین ایده‌ی نوآورانه، معماری‌ای ساخت که امروزه زیربنای تقریباً تمام مدل‌های زبانی بزرگ مثل GPT، BERT و خانواده‌ی آن‌هاست. نام این معماری ترنسفورمر است. در پایان این فصل خواهید توانست هر بخش از معماری ترنسفورمر را توضیح دهید، فرمول Multi-Head Attention را با مثال محاسبه کنید، نقش Positional Encoding را درک کنید، ساختار Encoder و Decoder ترنسفورمر را رسم و توضیح دهید، و اجزای کلیدی ترنسفورمر را در PyTorch از صفر پیاده‌سازی کنید.

---

# 🏗️ معماری کلی ترنسفورمر

ترنسفورمر اصلی از دو بخش اصلی تشکیل شده است: Encoder که ورودی را پردازش می‌کند و نمایش‌های غنی از آن می‌سازد، و Decoder که با استفاده از خروجی Encoder و آنچه تا الان تولید کرده، خروجی نهایی را گام‌به‌گام تولید می‌کند. هر Encoder از شش لایه‌ی یکسان (در ترنسفورمر اصلی) تشکیل شده و هر لایه دو زیربلوک دارد: یک Multi-Head Self-Attention و یک شبکه‌ی Feed-Forward دوطبقه. هر Decoder از شش لایه‌ی یکسان تشکیل شده که هر لایه سه زیربلوک دارد: Masked Multi-Head Self-Attention، Multi-Head Cross-Attention (توجه به خروجی Encoder)، و یک شبکه‌ی Feed-Forward.

در سراسر معماری، از دو تکنیک مهم استفاده می‌شود: اول، Residual Connection که خروجی هر زیربلوک را با ورودی آن جمع می‌کند تا جریان گرادیان حفظ شود؛ دقیقاً همان ایده‌ی ResNet. دوم، Layer Normalization که بعد از هر Residual Connection اعمال می‌شود و آموزش را پایدارتر می‌کند.

```
خروجی هر زیربلوک = LayerNorm( x + SubLayer(x) )
```

📷 [تصویر اینجا قرار گیرد: نمودار کامل معماری ترنسفورمر با نمایش Encoder سمت چپ (N لایه) و Decoder سمت راست (N لایه) با تمام اتصالات و زیربلوک‌ها]

---

# 🎭 Multi-Head Attention: چندین دیدگاه همزمان

در فصل قبل با Single-Head Attention آشنا شدیم. Multi-Head Attention ایده‌ی ساده‌ای دارد: به‌جای محاسبه‌ی یک توجه با ماتریس‌های Q، K، V با ابعاد کامل، چندین توجه موازی با ابعاد کوچک‌تر محاسبه کنیم و نتایج را کنار هم قرار دهیم. هر «سر» (Head) ممکن است روابط متفاوتی را یاد بگیرد؛ یک سر ممکن است به روابط نحوی توجه کند، سر دیگری به روابط معنایی، و سر سوم به الگوهای محلی. فرمول Multi-Head Attention به شکل زیر است:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O

head_i = Attention(Q × W_Q_i, K × W_K_i, V × W_V_i)
```

در این فرمول، h تعداد سرها است و W_Q_i، W_K_i، W_V_i ماتریس‌های وزن جداگانه برای هر سر هستند. اگر d_model بُعد مدل باشد و h تعداد سرها، بُعد هر سر برابر d_model / h خواهد بود که تعداد پارامترهای کل را ثابت نگه می‌دارد.

برای مثال، در ترنسفورمر اصلی d_model=512 و h=8 است. بُعد هر سر برابر 512/8=64 است. در هر گام، هشت توجه‌ی ۶۴ بعدی موازی محاسبه می‌شوند و خروجی‌های آن‌ها کنار هم قرار داده می‌شوند تا یک بردار ۵۱۲ بعدی (8×64) تشکیل شود که سپس از یک ماتریس پروجکشن نهایی W_O عبور می‌کند.

---

# 📍 Positional Encoding: زبان موقعیت

یکی از تفاوت‌های اساسی ترنسفورمر با RNN این است که Self-Attention ذاتاً از ترتیب عناصر بی‌اطلاع است. اگر ترتیب کلمات در یک جمله را عوض کنیم، Self-Attention بدون هیچ تغییری همان خروجی را می‌دهد (فقط با ردیف‌های جابه‌جاشده). برای رفع این مشکل، قبل از ورود توالی به ترنسفورمر، یک بردار موقعیت (Positional Encoding) به هر Embedding اضافه می‌شود.

در ترنسفورمر اصلی، این بردار با توابع سینوسی و کسینوسی محاسبه می‌شود:

```
PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )
```

در این فرمول، pos موقعیت کلمه در توالی است و i اندیس بُعد است. این روش دو ویژگی مهم دارد: اول، مقادیر آن برای هر موقعیت منحصربه‌فرد هستند. دوم، الگوهای ثابتی دارد که مدل می‌تواند روابط نسبی موقعیت‌ها را از آن استنباط کند.

---

# 🔢 مثال عددی Multi-Head Attention با دو سر

برای درک بهتر، یک Multi-Head Attention با h=2 و d_model=4 را با اعداد ساده حساب می‌کنیم. هر سر بُعد d_k = 4/2 = 2 خواهد داشت. فرض کنید توالی دو عنصری داریم:

ورودی X:
```
X = [[1, 0, 1, 0],
     [0, 1, 0, 1]]
```

ماتریس وزن سر اول برای Q، K، V (هرکدام 4×2):
```
W_Q1 = [[1, 0], [0, 1], [0, 0], [0, 0]]
W_K1 = [[1, 0], [0, 1], [0, 0], [0, 0]]
W_V1 = [[1, 0], [0, 1], [0, 0], [0, 0]]
```

Q1 = X × W_Q1:
```
Q1 = [[1×1+0×0+1×0+0×0, 1×0+0×1+1×0+0×0],
       [0×1+1×0+0×0+1×0, 0×0+1×1+0×0+1×0]]
Q1 = [[1, 0], [0, 1]]
```

به همین ترتیب K1 = [[1,0],[0,1]] و V1 = [[1,0],[0,1]].

محاسبه‌ی توجه سر اول:
```
Q1 × K1^T = [[1×1+0×0, 1×0+0×1],
               [0×1+1×0, 0×0+1×1]]
           = [[1, 0], [0, 1]]

تقسیم بر √2 ≈ 1.41: [[0.71, 0], [0, 0.71]]

Softmax ردیف اول: [e^0.71/(e^0.71+e^0), e^0/(e^0.71+e^0)]
                 = [2.03/3.03, 1.00/3.03] = [0.67, 0.33]
Softmax ردیف دوم: [0.33, 0.67]

head_1 = [[0.67×1+0.33×0, 0.67×0+0.33×1],
           [0.33×1+0.67×0, 0.33×0+0.67×1]]
       = [[0.67, 0.33], [0.33, 0.67]]
```

اگر سر دوم مستقل محاسبه شود و نتیجه‌اش head_2 باشد، خروجی نهایی از Concat(head_1, head_2) که یک ماتریس ۲×۴ است، از ماتریس W_O عبور می‌کند.

---

# 🔄 Feed-Forward Network در ترنسفورمر

بعد از هر Multi-Head Attention، یک شبکه‌ی Feed-Forward ساده قرار می‌گیرد که به هر موقعیت در توالی به‌طور مستقل اعمال می‌شود:

```
FFN(x) = max(0, x × W_1 + b_1) × W_2 + b_2
```

در ترنسفورمر اصلی، d_model=512 و بُعد لایه‌ی میانی ۲۰۴۸ است (چهار برابر d_model). این لایه‌ی FFN که پس از توجه قرار می‌گیرد، به مدل امکان می‌دهد اطلاعات توجه را به نمایش‌های پیچیده‌تر تبدیل کند.

---

# 💻 پروژه‌ی عملی فصل: پیاده‌سازی ترنسفورمر از صفر

در این پروژه تمام اجزای ترنسفورمر را از صفر پیاده‌سازی و برای یک مسئله‌ی طبقه‌بندی متن به کار می‌بریم.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset


# ---- ۱. Positional Encoding ----
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # ساخت ماتریس Positional Encoding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()

        # محاسبه‌ی div_term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        # بُعدهای زوج: sin، بُعدهای فرد: cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # اضافه کردن بُعد batch و ذخیره به‌عنوان buffer (نه پارامتر)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---- ۲. Multi-Head Attention ----
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # ماتریس‌های پروجکشن برای Q، K، V و خروجی
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        # (batch, seq, d_model) → (batch, heads, seq, d_k)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, Q_in, K_in, V_in, mask=None):
        batch_size = Q_in.size(0)

        # محاسبه‌ی Q، K، V و تقسیم به سرها
        Q = self.split_heads(self.W_Q(Q_in), batch_size)
        K = self.split_heads(self.W_K(K_in), batch_size)
        V = self.split_heads(self.W_V(V_in), batch_size)

        # محاسبه‌ی توجه
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = self.dropout(F.softmax(scores, dim=-1))
        attn_output = torch.matmul(weights, V)

        # ادغام سرها: (batch, heads, seq, d_k) → (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        return self.W_O(attn_output)


# ---- ۳. Feed-Forward Network ----
class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(
            self.dropout(F.relu(self.linear1(x)))
        )


# ---- ۴. Encoder Layer ----
class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Residual Connection + Self-Attention
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Residual Connection + Feed-Forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


# ---- ۵. مدل کامل ترنسفورمر برای طبقه‌بندی متن ----
class TransformerClassifier(nn.Module):

    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff, num_classes, max_seq_len, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def make_padding_mask(self, x):
        # ماسک برای جلوگیری از توجه به توکن‌های padding (اندیس 0)
        return (x != 0).unsqueeze(1).unsqueeze(2)

    def forward(self, x):
        mask = self.make_padding_mask(x)

        # Embedding + Positional Encoding
        out = self.pos_encoding(self.embedding(x))

        # عبور از لایه‌های Encoder
        for layer in self.encoder_layers:
            out = layer(out, mask)

        # Global Average Pooling و طبقه‌بندی
        pooled = out.mean(dim=1)
        return self.classifier(pooled)


# ---- آموزش و ارزیابی ----
vocab_size = 5000
max_seq_len = 64
d_model = 128
num_heads = 4
num_layers = 3
d_ff = 512
num_classes = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerClassifier(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    num_classes=num_classes,
    max_seq_len=max_seq_len
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"تعداد کل پارامترهای ترنسفورمر: {n_params:,}")

# داده‌ی ساختگی برای آزمایش
np.random.seed(42)
n_samples = 1000
X_data = torch.randint(1, vocab_size, (n_samples, max_seq_len))
# برچسب بر اساس میانگین اندیس‌های کلمات
y_data = (X_data.float().mean(dim=1) > vocab_size // 2).long()

# اضافه کردن Padding به ۳۰٪ آخر توالی‌ها برای واقعی‌تر کردن
for i in range(n_samples):
    pad_len = np.random.randint(0, max_seq_len // 3)
    if pad_len > 0:
        X_data[i, -pad_len:] = 0

X_train, y_train = X_data[:800], y_data[:800]
X_test, y_test = X_data[800:], y_data[800:]

trainloader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32, shuffle=True
)
testloader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=32, shuffle=False
)

optimizer = optim.AdamW(
    model.parameters(), lr=1e-4, weight_decay=1e-2
)
criterion = nn.CrossEntropyLoss()

# Warmup Scheduler: نرخ یادگیری ابتدا افزایش و سپس کاهش می‌یابد
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    steps_per_epoch=len(trainloader),
    epochs=20
)

print("\n=== آموزش ترنسفورمر ===\n")
best_acc = 0.0

for epoch in range(20):

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x_b, y_b in trainloader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        outputs = model(x_b)
        loss = criterion(outputs, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == y_b).sum().item()
        total += y_b.size(0)

    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for x_b, y_b in testloader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            out = model(x_b)
            test_correct += (out.argmax(1) == y_b).sum().item()
            test_total += y_b.size(0)

    train_acc = 100 * correct / total
    test_acc = 100 * test_correct / test_total
    if test_acc > best_acc:
        best_acc = test_acc

    if epoch % 4 == 0 or epoch == 19:
        print(f"دوره {epoch+1:2d} | خطا {total_loss/len(trainloader):.4f} | "
              f"آموزش {train_acc:.1f}% | آزمون {test_acc:.1f}% | "
              f"بهترین {best_acc:.1f}%")

print(f"\nبهترین دقت نهایی روی آزمون: {best_acc:.2f}%")
print("\nمقایسه‌ی پارامترهای اجزای مختلف:")
for name, module in model.named_children():
    n = sum(p.numel() for p in module.parameters())
    print(f"  {name}: {n:,} پارامتر")
```

این پیاده‌سازی چند نکته‌ی مهم دارد که باید به آن‌ها توجه کنید. نخست، در PositionalEncoding از `register_buffer` استفاده شده که یعنی این ماتریس با مدل ذخیره و بارگذاری می‌شود اما در آموزش به‌روز نمی‌شود. دوم، Padding Mask برای جلوگیری از توجه به توکن‌های padding استفاده شده؛ بدون این ماسک، مدل سعی می‌کند از اطلاعات padding که معنایی ندارند یاد بگیرد. سوم، از OneCycleLR Scheduler استفاده شده که یکی از بهترین روش‌های زمان‌بندی نرخ یادگیری برای ترنسفورمرهاست: ابتدا نرخ یادگیری را گرم (Warmup) می‌کند و سپس آن را به‌تدریج کاهش می‌دهد.

---

# ❓ سؤالات تستی

### سؤال ۱

چرا در Multi-Head Attention به‌جای یک توجه‌ی بزرگ، چندین توجه‌ی موازی با ابعاد کوچک‌تر استفاده می‌شود؟

الف) چون PyTorch از توجه‌ی تک‌سری بزرگ پشتیبانی نمی‌کند

ب) چون هر سر می‌تواند انواع مختلفی از روابط (نحوی، معنایی، محلی) را در زیرفضاهای مختلف یاد بگیرد که این تنوع نمایش‌ها قدرت بیانی مدل را افزایش می‌دهد

ج) چون Multi-Head Attention تعداد پارامترها را کاهش می‌دهد

د) چون توجه‌ی تک‌سری نمی‌تواند ماسک اعمال کند

### سؤال ۲

چرا در ترنسفورمر Positional Encoding ضروری است در حالی که در RNN نیازی به آن نیست؟

الف) چون ترنسفورمر پارامتر بیشتری دارد

ب) چون Self-Attention ذاتاً از ترتیب عناصر توالی بی‌اطلاع است و بدون اطلاعات موقعیت، جابه‌جایی ترتیب کلمات روی خروجی تأثیر نمی‌گذارد؛ اما در RNN اطلاعات ترتیبی ذاتاً در ساختار پردازش گام‌به‌گام وجود دارد

ج) چون ترنسفورمر از Embedding استفاده می‌کند

د) چون Positional Encoding یادگیری را سریع‌تر می‌کند

---

## ✅ پاسخ‌ها

**پاسخ سؤال ۱: گزینه ب**

هر سر از Multi-Head Attention در یک زیرفضای متفاوت از بردارهای Q، K، V کار می‌کند. این یعنی هر سر می‌تواند روابط متفاوتی بین کلمات را یاد بگیرد. به‌عنوان مثال، در یک مدل زبانی آموزش‌دیده، دیده شده که بعضی سرها به وابستگی‌های نحوی توجه می‌کنند، بعضی به وابستگی‌های معنایی، و بعضی به الگوهای موقعیتی. این تنوع دیدگاه‌ها قدرت بیانی مدل را به‌طور قابل‌توجهی افزایش می‌دهد.

**پاسخ سؤال ۲: گزینه ب**

وقتی Self-Attention امتیاز توجه را بین Q و K محاسبه می‌کند، تنها محتوای هر موقعیت مهم است، نه مکان آن. جمله‌ی «سگ گربه را دید» و «گربه سگ را دید» بدون Positional Encoding کاملاً یکسان دیده می‌شوند. اما RNN به‌طور ذاتی ترتیبی است و گام اول قبل از گام دوم پردازش می‌شود، پس نیازی به کدگذاری موقعیت ندارد.

---

# 📝 خلاصه فصل

در این فصل معماری کامل ترنسفورمر را بررسی کردیم. Multi-Head Attention را به‌عنوان توسعه‌ی Self-Attention برای یادگیری روابط متنوع آموختیم و فرمول آن را با مثال عددی محاسبه کردیم. Positional Encoding با توابع سینوسی/کسینوسی را درک کردیم و فهمیدیم چرا ترنسفورمر برخلاف RNN به آن نیاز دارد. ساختار EncoderLayer با Residual Connection و LayerNorm را بررسی کردیم. در پروژه‌ی عملی تمام این اجزا را از صفر در PyTorch پیاده‌سازی کردیم و یک ترنسفورمر کامل برای طبقه‌بندی متن ساختیم.

---

# 🎯 تمرین‌های پایان فصل

۱. یک DecoderLayer کامل با Masked Self-Attention و Cross-Attention پیاده‌سازی کنید.

۲. Positional Encoding قابل‌یادگیری (Learnable) را به‌جای ثابت پیاده‌سازی کنید و عملکرد آن را با نسخه‌ی سینوسی مقایسه کنید.

۳. تعداد سرها را از ۴ به ۸ افزایش دهید و ببینید تأثیر آن بر دقت و زمان آموزش چیست.

۴. مدل ترنسفورمر این فصل را روی دیتاست IMDB آموزش دهید و با LSTM مقایسه کنید.

۵. نمودار نرخ یادگیری OneCycleLR را در طول آموزش رسم کنید و مرحله‌ی Warmup را مشاهده کنید.

---

در فصل ۲۱ با **پیش‌پردازش متن و بازنمایی کلمات** آشنا می‌شویم؛ از Tokenization و حذف کلمات توقف تا Word2Vec و GloVe که پایه‌ی درک مدل‌های زبانی مدرن هستند.
