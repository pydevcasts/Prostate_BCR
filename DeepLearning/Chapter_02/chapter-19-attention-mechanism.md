# فصل ۱۹ 👁️ مکانیزم توجه (Attention Mechanism)

## 🎯 اهداف فصل

در بخش سوم کتاب با شبکه‌های بازگشتی آشنا شدیم که می‌توانند وابستگی‌های زمانی را مدل کنند. اما این معماری‌ها یک محدودیت اساسی دارند: اطلاعات کل توالی باید در یک بردار حالت پنهان با اندازه‌ی ثابت خلاصه شود. تصور کنید می‌خواهید یک جمله‌ی صد کلمه‌ای را ترجمه کنید؛ خلاصه کردن تمام آن اطلاعات در یک بردار ۲۵۶ بعدی اطلاعات زیادی را از دست می‌دهد. در سال ۲۰۱۴، بادانائو و همکارانش مکانیزمی پیشنهاد دادند که این مشکل را به‌طور اساسی حل می‌کند: مکانیزم توجه (Attention Mechanism). در پایان این فصل خواهید توانست محدودیت گلوگاه اطلاعات در RNN را توضیح دهید، مفهوم توجه را با تمثیل‌های شهودی درک کنید، فرمول‌های Additive Attention و Scaled Dot-Product Attention را با مثال عددی محاسبه کنید، Self-Attention را پیاده‌سازی کنید، و تفاوت ماهوی معماری مبتنی بر توجه با RNN را بشناسید.

---

# 🔍 مشکل گلوگاه اطلاعات در RNN

در یک معماری Seq2Seq (دنباله به دنباله) مبتنی بر RNN، کل متن ورودی باید در حالت پنهان آخرین گام Encoder خلاصه شود. این حالت پنهان که بردار زمینه (Context Vector) نامیده می‌شود، تنها اطلاعاتی است که Decoder برای تولید خروجی دارد. در جملات کوتاه این رویکرد نسبتاً خوب کار می‌کند، اما برای جملات طولانی‌تر، بردار زمینه‌ی ثابت‌اندازه نمی‌تواند تمام اطلاعات لازم را در خود جای دهد. همچنین کلماتی که در ابتدای جمله آمده‌اند، اطلاعاتشان در طول پردازش جمله تضعیف می‌شود.

یک انسان هنگام ترجمه این‌گونه عمل نمی‌کند؛ وقتی می‌خواهد کلمه‌ای را ترجمه کند، نگاهش مستقیماً به بخش مرتبط جمله‌ی اصلی برمی‌گردد. مکانیزم توجه دقیقاً همین رفتار را شبیه‌سازی می‌کند: در هر گام تولید خروجی، مدل به تمام حالت‌های پنهان Encoder دسترسی مستقیم دارد و با محاسبه‌ی یک توزیع وزنی (توزیع توجه) تعیین می‌کند که در این لحظه باید به کدام بخش‌های ورودی بیشتر «توجه» کند.

📷 [تصویر اینجا قرار گیرد: نمودار مقایسه‌ی معماری Seq2Seq بدون توجه (فقط یک بردار زمینه) در برابر Seq2Seq با Attention (خطوط وزن‌دار از تمام حالت‌های Encoder به هر گام Decoder)]

---

# 💡 ایده‌ی شهودی توجه

قبل از ورود به ریاضیات، یک تمثیل مفید را بررسی می‌کنیم. فرض کنید در یک کتابخانه هستید و می‌خواهید اطلاعاتی درباره‌ی یک موضوع پیدا کنید. شما یک سؤال (Query) دارید، کتابخانه مجموعه‌ای از کتاب‌ها با برچسب (Key) دارد، و هر کتاب محتوایی (Value) دارد. شما ابتدا سؤال خود را با برچسب هر کتاب مقایسه می‌کنید تا مرتبط‌ترین‌ها را بیابید، سپس محتوای کتاب‌های مرتبط را با وزن‌های متناسب با ارتباطشان ترکیب می‌کنید.

مکانیزم توجه دقیقاً همین کار را به‌صورت ریاضی انجام می‌دهد. وقتی Decoder می‌خواهد کلمه‌ای تولید کند، یک Query دارد (حالت پنهان فعلی Decoder)، حالت‌های پنهان Encoder به‌عنوان Key و Value عمل می‌کنند، و مدل یاد می‌گیرد که چگونه Query را با هر Key مقایسه کند تا وزن توجه مناسب را محاسبه کند.

---

# 📐 Additive Attention (Bahdanau Attention)

اولین فرمول‌بندی رسمی توجه که در سال ۲۰۱۴ معرفی شد، توجه جمعی (Additive Attention) نام دارد. در این روش، امتیاز توجه بین یک حالت پنهان Decoder (s) و هر حالت پنهان Encoder (h_i) با یک شبکه‌ی عصبی کوچک محاسبه می‌شود:

```
score(s, h_i) = v_a^T × tanh( W_a × s + U_a × h_i )
```

در این فرمول، W_a، U_a و v_a پارامترهای قابل‌یادگیری شبکه هستند. سپس این امتیازها از Softmax عبور می‌کنند تا یک توزیع احتمال (وزن‌های توجه) تشکیل دهند:

```
α_i = softmax( score(s, h_i) ) = exp(score(s, h_i)) / Σ_j exp(score(s, h_j))
```

و بردار زمینه‌ی پویا (Dynamic Context Vector) از ترکیب وزن‌دار تمام حالت‌های Encoder محاسبه می‌شود:

```
c = Σ_i α_i × h_i
```

این بردار c برای هر گام تولید متفاوت است و دقیقاً به بخش‌هایی از ورودی که برای تولید این گام مرتبط هستند، وزن بیشتری می‌دهد.

---

# ⚡ Scaled Dot-Product Attention

در سال ۲۰۱۷، مقاله‌ی «Attention is All You Need» نوع دیگری از توجه را معرفی کرد که هم ساده‌تر است و هم بسیار کارآمدتر از Additive Attention. این روش که Scaled Dot-Product Attention نامیده می‌شود، پایه‌ی معماری ترنسفورمر است که در فصل بعدی به‌طور کامل بررسی می‌شود.

در این روش، سه ماتریس Query (Q)، Key (K) و Value (V) داریم که هرکدام از ضرب ورودی در یک ماتریس وزن مربوطه به دست می‌آیند:

```
Q = X × W_Q
K = X × W_K
V = X × W_V
```

سپس توجه به شکل زیر محاسبه می‌شود:

```
Attention(Q, K, V) = softmax( Q × K^T / √d_k ) × V
```

در این فرمول، d_k اندازه‌ی بُعد Key است و تقسیم بر √d_k یک عامل مقیاس‌بندی است. بدون این مقیاس‌بندی، وقتی d_k بزرگ باشد، حاصل‌ضرب Q × K^T مقادیر بسیار بزرگی می‌گیرد که بعد از Softmax به توزیع‌هایی بسیار تیز تبدیل می‌شوند و گرادیان‌ها محو می‌شوند.

---

# 🔢 مثال عددی کامل Scaled Dot-Product Attention

برای درک عینی این فرمول، یک مثال با اعداد ساده محاسبه می‌کنیم. فرض کنید یک توالی‌ی سه کلمه‌ای داریم و d_k = 2:

ماتریس Query (۳×۲):
```
Q = [[1, 0],
     [0, 1],
     [1, 1]]
```

ماتریس Key (۳×۲):
```
K = [[1, 1],
     [0, 1],
     [1, 0]]
```

ماتریس Value (۳×۲):
```
V = [[1, 2],
     [3, 4],
     [5, 6]]
```

محاسبه‌ی Q × K^T (ماتریس ۳×۳ امتیازهای توجه):
```
Q × K^T:
ردیف ۱: [1×1+0×1=1,  1×0+0×1=0,  1×1+0×0=1]
ردیف ۲: [0×1+1×1=1,  0×0+1×1=1,  0×1+1×0=0]
ردیف ۳: [1×1+1×1=2,  1×0+1×1=1,  1×1+1×0=1]

نتیجه:
[[1, 0, 1],
 [1, 1, 0],
 [2, 1, 1]]
```

تقسیم بر √d_k = √2 ≈ 1.41:
```
[[0.71, 0.00, 0.71],
 [0.71, 0.71, 0.00],
 [1.41, 0.71, 0.71]]
```

اعمال Softmax روی هر ردیف (ردیف اول):
```
exp(0.71)≈2.03, exp(0.00)≈1.00, exp(0.71)≈2.03
مجموع: 5.06
وزن‌ها: [0.401, 0.198, 0.401]
```

ضرب وزن‌ها در Value برای ردیف اول:
```
0.401 × [1,2] + 0.198 × [3,4] + 0.401 × [5,6]
= [0.401, 0.802] + [0.594, 0.792] + [2.005, 2.406]
= [3.000, 4.000]
```

این بردار [3.0, 4.0] خروجی توجه برای کلمه‌ی اول است که نشان می‌دهد مدل به کلمه‌ی اول و سوم به‌طور برابر (و بیشتر از کلمه‌ی دوم) توجه کرده است.

---

# 🔄 Self-Attention: توجه به خود توالی

یکی از قوی‌ترین کاربردهای توجه، Self-Attention است که در آن Q، K و V همه از یک توالی می‌آیند. این به مدل اجازه می‌دهد روابط بین هر کلمه با تمام کلمات دیگر در همان جمله را بیاموزد. به‌عنوان مثال، در جمله‌ی «بانک رودخانه خروشان بود»، Self-Attention می‌تواند یاد بگیرد که «بانک» باید بیشتر به «رودخانه» توجه کند تا معنای درست آن را بفهمد.

📷 [تصویر اینجا قرار گیرد: نمودار Self-Attention روی یک جمله که با خطوط وزن‌دار نشان می‌دهد چگونه هر کلمه به کلمات دیگر توجه می‌کند]

---

# 💻 پروژه‌ی عملی فصل: پیاده‌سازی Self-Attention از صفر

در این پروژه Self-Attention و Scaled Dot-Product Attention را از صفر پیاده‌سازی می‌کنیم و سپس آن را به‌عنوان یک لایه در یک مدل طبقه‌بندی متن استفاده می‌کنیم.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# ---- پیاده‌سازی دستی Scaled Dot-Product Attention ----
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    پیاده‌سازی Attention(Q,K,V) = softmax(QK^T/√d_k) × V

    ورودی‌ها:
        Q: (batch, seq_q, d_k)
        K: (batch, seq_k, d_k)
        V: (batch, seq_k, d_v)
    خروجی:
        output: (batch, seq_q, d_v)
        weights: (batch, seq_q, seq_k)
    """

    d_k = Q.size(-1)

    # محاسبه‌ی امتیاز توجه
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    # اعمال Mask در صورت نیاز (برای Padding یا توجه علّی)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax برای تبدیل به توزیع احتمال
    weights = F.softmax(scores, dim=-1)

    # ضرب وزن‌دار در Value
    output = torch.matmul(weights, V)

    return output, weights


class SelfAttentionLayer(nn.Module):
    """
    لایه‌ی Self-Attention با ماتریس‌های وزن قابل‌یادگیری W_Q، W_K، W_V
    """

    def __init__(self, d_model, d_k=None):
        super().__init__()
        d_k = d_k or d_model

        # ماتریس‌های تبدیل ورودی به Q، K، V
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_k, bias=False)

        # لایه‌ی پروجکشن نهایی
        self.W_O = nn.Linear(d_k, d_model, bias=False)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        output, weights = scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_O(output)

        return output, weights


# ---- آزمایش پیاده‌سازی دستی ----
print("=== آزمایش پیاده‌سازی Self-Attention ===\n")

# مثال عددی با همان اعداد توضیحات فصل
Q_test = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]])
K_test = torch.tensor([[[1., 1.], [0., 1.], [1., 0.]]])
V_test = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]])

output, weights = scaled_dot_product_attention(Q_test, K_test, V_test)
print("وزن‌های توجه برای هر موقعیت:")
for i in range(3):
    print(f"  موقعیت {i+1}: {weights[0, i].tolist()}")
print(f"\nخروجی توجه:")
for i in range(3):
    print(f"  موقعیت {i+1}: {output[0, i].tolist()}")


# ---- مدل طبقه‌بندی متن با Self-Attention ----
class TextClassifierWithAttention(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_heads_equiv,
                 num_classes, max_seq_len, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Embedding(max_seq_len, embed_dim)

        # چند لایه Self-Attention موازی (شبیه Multi-Head اما ساده‌شده)
        self.attention_layers = nn.ModuleList([
            SelfAttentionLayer(embed_dim)
            for _ in range(num_heads_equiv)
        ])

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len = x.shape
        device = x.device

        # Embedding + Positional Encoding
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        embedded = self.embedding(x) + self.pos_encoding(positions)
        embedded = self.dropout(embedded)

        # اعمال Self-Attention و ترکیب خروجی‌ها
        attention_outputs = []
        for attn_layer in self.attention_layers:
            attn_out, _ = attn_layer(embedded)
            attention_outputs.append(attn_out)

        # میانگین خروجی لایه‌های توجه + Residual Connection
        combined = sum(attention_outputs) / len(attention_outputs)
        out = self.layer_norm(embedded + combined)

        # Global Average Pooling در بُعد توالی
        pooled = out.mean(dim=1)

        return self.classifier(pooled)


# ---- دیتاست ساختگی برای آزمایش ----
vocab_size = 1000
max_seq_len = 50
num_classes = 2
batch_size = 32

def generate_dummy_text_data(n_samples=500):
    """
    تولید داده‌ی ساختگی برای تست مدل توجه:
    اگر ابتدای توالی عدد کمتر از ۵۰۰ باشد، برچسب ۰
    وگرنه برچسب ۱
    """
    X = torch.randint(1, vocab_size, (n_samples, max_seq_len))
    y = (X[:, 0] > 500).long()
    return X, y

X, y = generate_dummy_text_data(500)
X_train, y_train = X[:400], y[:400]
X_test, y_test = X[400:], y[400:]

from torch.utils.data import TensorDataset, DataLoader

train_loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=32, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(X_test, y_test), batch_size=32, shuffle=False
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TextClassifierWithAttention(
    vocab_size=vocab_size,
    embed_dim=64,
    num_heads_equiv=2,
    num_classes=num_classes,
    max_seq_len=max_seq_len
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nتعداد پارامترها: {n_params:,}")

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == y_batch).sum().item()
        total += y_batch.size(0)

    if epoch % 5 == 0 or epoch == 19:
        train_acc = 100 * correct / total
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                out = model(x_b)
                test_correct += (out.argmax(1) == y_b).sum().item()
                test_total += y_b.size(0)
        test_acc = 100 * test_correct / test_total
        print(f"دوره {epoch+1:2d} | آموزش: {train_acc:.1f}% | آزمون: {test_acc:.1f}%")

# ---- تجسم وزن‌های توجه ----
def visualize_attention_weights(model, text_indices, device):
    """
    نمایش وزن‌های توجه برای یک نمونه‌ی متنی
    این تجسم نشان می‌دهد مدل به کدام بخش‌های توالی بیشتر توجه کرده
    """

    model.eval()
    x = torch.tensor([text_indices]).to(device)

    with torch.no_grad():
        embedded = model.embedding(x) + model.pos_encoding(
            torch.arange(len(text_indices), device=device).unsqueeze(0)
        )
        _, weights = model.attention_layers[0](embedded)

    attention_matrix = weights[0].cpu().numpy()

    print("\nماتریس توجه (ردیف=موقعیت فعلی، ستون=موقعیت مورد توجه):")
    print("هر عدد نشان می‌دهد موقعیت ردیف چقدر به موقعیت ستون توجه کرده:")
    print(np.round(attention_matrix[:5, :5], 3))


sample_indices = list(range(1, max_seq_len + 1))
visualize_attention_weights(model, sample_indices, device)
```

---

# ❓ سؤالات تستی

### سؤال ۱

چرا در Scaled Dot-Product Attention، حاصل‌ضرب Q × K^T بر √d_k تقسیم می‌شود؟

الف) برای افزایش سرعت محاسبه

ب) چون وقتی d_k بزرگ باشد، مقادیر حاصل‌ضرب خیلی بزرگ می‌شوند و Softmax را به توزیع‌های بسیار تیز تبدیل می‌کنند که باعث محوشدگی گرادیان می‌شود

ج) چون تقسیم بر √d_k وزن‌های توجه را برابر می‌کند

د) چون این روش در کد PyTorch پیش‌فرض است

### سؤال ۲

Self-Attention در مقایسه با RNN چه مزیت اصلی دارد؟

الف) Self-Attention پارامتر کمتری دارد

ب) Self-Attention می‌تواند به‌صورت موازی پردازش کند و روابط مستقیم بین هر دو موقعیت دلخواه در توالی را در یک گام یاد بگیرد، در حالی که RNN باید این روابط را از طریق گام‌های متوالی عبور دهد

ج) Self-Attention برای داده‌های عددی بهتر است

د) Self-Attention نیازی به آموزش ندارد

---

## ✅ پاسخ‌ها

**پاسخ سؤال ۱: گزینه ب**

وقتی d_k بزرگ است، مقادیر Q × K^T به مقیاس √d_k رشد می‌کنند. بدون مقیاس‌بندی، این مقادیر بزرگ در Softmax به توزیع‌های تیزی تبدیل می‌شوند که شبیه تابع argmax رفتار می‌کنند و گرادیان‌ها تقریباً صفر می‌شوند. تقسیم بر √d_k این مقادیر را در دامنه‌ای نگه می‌دارد که Softmax بهتر کار می‌کند.

**پاسخ سؤال ۲: گزینه ب**

در RNN، برای اینکه اطلاعات کلمه‌ی اول به کلمه‌ی صدم برسد، باید از ۹۹ گام متوالی عبور کند که احتمال محوشدگی را بالا می‌برد. در Self-Attention، هر دو موقعیت دلخواه مستقیماً با هم مقایسه می‌شوند (فاصله‌ی O(1)) و این محاسبات می‌توانند برای تمام جفت موقعیت‌ها به‌صورت موازی انجام شوند که سرعت آموزش را به‌طور چشمگیری افزایش می‌دهد.

---

# 📝 خلاصه فصل

در این فصل با مکانیزم توجه به‌عنوان یکی از مهم‌ترین نوآوری‌های دهه‌ی اخیر در یادگیری عمیق آشنا شدیم. دیدیم که توجه مشکل گلوگاه اطلاعات در RNN را با دادن دسترسی مستقیم Decoder به تمام حالت‌های Encoder حل می‌کند. Additive Attention و Scaled Dot-Product Attention را با فرمول کامل و مثال عددی گام‌به‌گام بررسی کردیم. مفهوم Self-Attention را درک کردیم که می‌تواند روابط بین هر دو موقعیت در یک توالی را در یک گام و به‌صورت موازی یاد بگیرد. در پروژه‌ی عملی Self-Attention را از صفر پیاده‌سازی کردیم و آن را در یک مدل طبقه‌بندی متن به کار بردیم.

---

# 🎯 تمرین‌های پایان فصل

۱. مثال عددی این فصل را برای ردیف‌های دوم و سوم ماتریس Q هم کامل کنید.

۲. یک Causal Mask (Mask علّی) به پیاده‌سازی Self-Attention اضافه کنید تا موقعیت i نتواند به موقعیت‌های بعد از خود توجه کند.

۳. Multi-Head Attention را پیاده‌سازی کنید که چندین Self-Attention موازی اجرا می‌کند.

۴. مدل طبقه‌بندی را روی دیتاست IMDB آزمایش کنید و با LSTM مقایسه کنید.

۵. وزن‌های توجه مدل آموزش‌دیده را روی چند نمونه‌ی واقعی تجسم کنید.

---

در فصل ۲۰ با **معماری ترنسفورمر** آشنا می‌شویم که با ترکیب Multi-Head Attention، Positional Encoding و Feed-Forward Network، یکی از تأثیرگذارترین معماری‌های تاریخ یادگیری عمیق را تشکیل می‌دهد.
