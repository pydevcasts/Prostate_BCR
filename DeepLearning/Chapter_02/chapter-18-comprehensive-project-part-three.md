# فصل ۱۸ 📈 پروژه‌ی جامع بخش سوم: پیش‌بینی سری زمانی مالی و تولید متن

## 🎯 اهداف فصل

این فصل پروژه‌ی جامع بخش سوم کتاب است و تمام آنچه در فصل‌های ۱۴ تا ۱۷ آموختیم را در دو پروژه‌ی متفاوت و واقعی به کار می‌بریم. پروژه‌ی اول پیش‌بینی قیمت سهام با LSTM است که نمونه‌ای کلاسیک از پیش‌بینی سری زمانی مالی است. پروژه‌ی دوم تولید متن شعرگونه با GRU است که یک مسئله‌ی تولیدی جذاب است که نشان می‌دهد همان معماری‌های بازگشتی می‌توانند هم برای پیش‌بینی و هم برای تولید استفاده شوند. در پایان این فصل خواهید توانست یک خط لوله‌ی کامل پیش‌بینی سری زمانی مالی بسازید، مفهوم تولید متن با شبکه‌های بازگشتی را درک و پیاده‌سازی کنید، و تفاوت مسائل پیش‌بینی (Prediction) و مسائل تولید (Generation) را در زمینه‌ی داده‌های ترتیبی درک کنید.

---

# 📊 پروژه‌ی اول: پیش‌بینی قیمت سهام با LSTM

پیش‌بینی قیمت سهام یکی از کاربردهای رایج شبکه‌های بازگشتی در صنعت مالی است. البته باید صادقانه گفت که بازارهای مالی بسیار پیچیده‌اند و هیچ مدلی نمی‌تواند با قطعیت آینده را پیش‌بینی کند. هدف این پروژه نه ساخت یک ابزار معاملاتی سودآور، بلکه نشان دادن نحوه‌ی ساخت یک خط لوله‌ی کامل پیش‌بینی سری زمانی با چندین ویژگی (Multi-variate) است که در پروژه‌های صنعتی واقعی رایج است.

ما از دیتاست قیمت سهام اپل (AAPL) استفاده می‌کنیم که شامل قیمت باز، بسته، بالا، پایین و حجم معاملات روزانه است. به‌جای پیش‌بینی از روی یک ویژگی (مثل فصل ۱۴)، اینجا از همه‌ی این ویژگی‌ها با هم برای پیش‌بینی قیمت بسته‌شدن (Close Price) استفاده می‌کنیم.

### مرحله‌ی اول: بارگذاری و آماده‌سازی داده

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset, DataLoader

# دانلود داده‌ی سهام اپل با yfinance
# pip install yfinance
try:
    import yfinance as yf
    df = yf.download('AAPL', start='2018-01-01', end='2023-12-31')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    print(f"داده از Yahoo Finance بارگذاری شد: {len(df)} روز معاملاتی")
except:
    # اگر yfinance نصب نیست، داده‌ی ساختگی می‌سازیم
    print("yfinance در دسترس نیست. استفاده از داده‌ی شبیه‌سازی‌شده...")
    np.random.seed(42)
    n = 1500
    t = np.linspace(0, 4*np.pi, n)
    close = 150 + 30*np.sin(t) + np.random.randn(n)*5 + np.linspace(0, 50, n)
    df = pd.DataFrame({
        'Open': close * (1 + np.random.randn(n)*0.005),
        'High': close * (1 + abs(np.random.randn(n)*0.01)),
        'Low':  close * (1 - abs(np.random.randn(n)*0.01)),
        'Close': close,
        'Volume': (1e7 * (1 + np.random.randn(n)*0.3)).astype(int)
    })

features = ['Open', 'High', 'Low', 'Close', 'Volume']
target_idx = features.index('Close')

# نرمال‌سازی مستقل برای هر ویژگی
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[features].values).astype(np.float32)

print(f"\nشکل داده: {data_scaled.shape}")
print(f"ویژگی‌ها: {features}")
print(f"ویژگی هدف: Close (اندیس {target_idx})")


# ---- دیتاست چندویژگی ----
class StockDataset(Dataset):

    def __init__(self, data, seq_len, target_idx):
        self.X = []
        self.y = []
        for i in range(len(data) - seq_len):
            self.X.append(torch.tensor(data[i:i+seq_len]))
            self.y.append(torch.tensor(data[i+seq_len, target_idx]))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


seq_len = 30
split = int(len(data_scaled) * 0.8)

train_data = data_scaled[:split + seq_len]
test_data = data_scaled[split:]

train_dataset = StockDataset(train_data, seq_len, target_idx)
test_dataset = StockDataset(test_data, seq_len, target_idx)

trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"\nنمونه‌های آموزش: {len(train_dataset)}")
print(f"نمونه‌های آزمون: {len(test_dataset)}")
```

### مرحله‌ی دوم: مدل LSTM چندویژگی

```python
class StockLSTM(nn.Module):

    def __init__(self, input_size=5, hidden_size=128, num_layers=3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # Attention Layer برای وزن‌دهی به گام‌های مختلف زمانی
        self.attention = nn.Linear(hidden_size, 1)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # Attention: محاسبه‌ی وزن اهمیت هر گام زمانی
        attn_weights = torch.softmax(
            self.attention(lstm_out), dim=1
        )
        context = (attn_weights * lstm_out).sum(dim=1)

        return self.regressor(context).squeeze(1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StockLSTM().to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nتعداد پارامترها: {n_params:,}")

criterion = nn.HuberLoss()  # مقاوم‌تر در برابر مقادیر پرت نسبت به MSE
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

best_loss = float('inf')
best_weights = None

for epoch in range(50):

    model.train()
    train_loss = 0.0
    for x_b, y_b in trainloader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        preds = model(x_b)
        loss = criterion(preds, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_b, y_b in testloader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            val_loss += criterion(model(x_b), y_b).item()

    val_loss /= len(testloader)
    if val_loss < best_loss:
        best_loss = val_loss
        best_weights = {k: v.clone() for k, v in model.state_dict().items()}

    scheduler.step()

    if epoch % 10 == 0 or epoch == 49:
        print(f"دوره {epoch+1:2d} | آموزش: {train_loss/len(trainloader):.6f} | "
              f"آزمون: {val_loss:.6f}")

# ارزیابی نهایی
model.load_state_dict(best_weights)
model.eval()

all_preds, all_targets = [], []
with torch.no_grad():
    for x_b, y_b in testloader:
        preds = model(x_b.to(device)).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y_b.numpy())

# تبدیل به مقیاس اصلی
dummy = np.zeros((len(all_preds), len(features)))
dummy[:, target_idx] = all_preds
preds_orig = scaler.inverse_transform(dummy)[:, target_idx]

dummy[:, target_idx] = all_targets
targets_orig = scaler.inverse_transform(dummy)[:, target_idx]

mae = mean_absolute_error(targets_orig, preds_orig)
rmse = np.sqrt(mean_squared_error(targets_orig, preds_orig))

print(f"\nنتایج نهایی پروژه‌ی اول:")
print(f"MAE: {mae:.2f} دلار")
print(f"RMSE: {rmse:.2f} دلار")
```

---

# ✍️ پروژه‌ی دوم: تولید متن با GRU

در پروژه‌ی دوم از GRU برای یادگیری سبک یک متن و تولید متن جدید در همان سبک استفاده می‌کنیم. این نوع مدل را مدل زبانی مبتنی بر کاراکتر (Character-level Language Model) می‌نامند زیرا در سطح کاراکتر (نه کلمه) کار می‌کند. مدل یاد می‌گیرد که بعد از هر دنباله‌ای از کاراکترها، چه کاراکتری احتمال بیشتری دارد که بیاید و در زمان تولید، این احتمالات را برای ساختن متن جدید استفاده می‌کند.

```python
# ---- پروژه‌ی دوم: تولید متن با GRU ----

# متن نمونه برای آموزش (می‌توانید متن دلخواه بدهید)
sample_text = """
deep learning is a type of machine learning that uses neural networks
with many layers to learn from large amounts of data
neural networks are inspired by the human brain and can learn complex patterns
recurrent neural networks are designed for sequential data
long short term memory networks solve the vanishing gradient problem
gated recurrent units are simpler than lstm but equally powerful
transformers have revolutionized natural language processing
attention mechanisms allow models to focus on relevant parts of input
""" * 20  # تکرار برای ایجاد داده کافی

# ---- ساخت واژگان کاراکتری ----
chars = sorted(set(sample_text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)

print(f"اندازه واژگان: {vocab_size} کاراکتر منحصربه‌فرد")
print(f"طول کل متن: {len(sample_text)} کاراکتر")

# تبدیل متن به اندیس
text_indices = [char_to_idx[c] for c in sample_text]

# ---- دیتاست کاراکتری ----
class CharDataset(Dataset):

    def __init__(self, text_indices, seq_len):
        self.X = []
        self.y = []
        for i in range(0, len(text_indices) - seq_len - 1, 3):
            self.X.append(torch.tensor(
                text_indices[i:i+seq_len], dtype=torch.long
            ))
            self.y.append(torch.tensor(
                text_indices[i+1:i+seq_len+1], dtype=torch.long
            ))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

seq_len = 50
char_dataset = CharDataset(text_indices, seq_len)
char_loader = DataLoader(char_dataset, batch_size=64, shuffle=True)

print(f"تعداد نمونه‌ها: {len(char_dataset)}")

# ---- مدل GRU برای تولید متن ----
class CharGRU(nn.Module):

    def __init__(self, vocab_size, embed_dim=64, hidden_size=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            embed_dim, hidden_size, num_layers,
            batch_first=True, dropout=0.3
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        out, hidden = self.gru(embedded, hidden)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size
        ).to(device)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
char_model = CharGRU(vocab_size).to(device)

n_params = sum(p.numel() for p in char_model.parameters() if p.requires_grad)
print(f"تعداد پارامترها: {n_params:,}")

criterion_char = nn.CrossEntropyLoss()
optimizer_char = optim.Adam(char_model.parameters(), lr=0.003)

# ---- آموزش مدل تولید متن ----
for epoch in range(30):

    char_model.train()
    total_loss = 0.0

    for x_batch, y_batch in char_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer_char.zero_grad()
        logits, _ = char_model(x_batch)

        # logits: (batch, seq_len, vocab_size)، y: (batch, seq_len)
        loss = criterion_char(
            logits.reshape(-1, vocab_size),
            y_batch.reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(char_model.parameters(), 1.0)
        optimizer_char.step()
        total_loss += loss.item()

    if epoch % 5 == 0 or epoch == 29:
        perplexity = np.exp(total_loss / len(char_loader))
        print(f"دوره {epoch+1:2d} | خطا: {total_loss/len(char_loader):.4f} | "
              f"Perplexity: {perplexity:.2f}")


# ---- تابع تولید متن ----
def generate_text(model, seed_text, length=200, temperature=0.8, device='cpu'):
    """
    تولید متن با استفاده از مدل GRU آموزش‌دیده.
    temperature پارامتر خلاقیت است:
    - مقدار کم (0.3): متن قابل‌پیش‌بینی‌تر
    - مقدار زیاد (1.5): متن خلاقانه‌تر اما احتمالاً نامفهوم‌تر
    """

    model.eval()
    chars = list(seed_text)
    hidden = model.init_hidden(1, device)

    # گرم‌کردن مدل با متن اولیه
    with torch.no_grad():
        for char in seed_text[:-1]:
            if char not in char_to_idx:
                continue
            x = torch.tensor([[char_to_idx[char]]]).to(device)
            _, hidden = model(x, hidden)

        # تولید کاراکترهای جدید
        current_char = seed_text[-1]
        for _ in range(length):
            if current_char not in char_to_idx:
                current_char = ' '
            x = torch.tensor([[char_to_idx[current_char]]]).to(device)
            logits, hidden = model(x, hidden)

            # اعمال temperature برای کنترل خلاقیت
            probs = torch.softmax(logits[0, 0] / temperature, dim=0)
            next_idx = torch.multinomial(probs, 1).item()
            current_char = idx_to_char[next_idx]
            chars.append(current_char)

    return ''.join(chars)


print("\n=== نتایج تولید متن ===\n")

for temp in [0.5, 0.8, 1.2]:
    print(f"Temperature = {temp}:")
    generated = generate_text(
        char_model, "deep learning", length=150,
        temperature=temp, device=device
    )
    print(generated)
    print("-" * 50)
```

پارامتر temperature در تابع `generate_text` نقش بسیار مهمی دارد. وقتی temperature کوچک است (مثل ۰.۳)، توزیع احتمال خروجی بسیار تیز می‌شود و مدل تقریباً همیشه محتمل‌ترین کاراکتر بعدی را انتخاب می‌کند؛ متن حاصل قابل‌پیش‌بینی‌تر و دستوری‌تر است اما ممکن است تکراری باشد. وقتی temperature بزرگ است (مثل ۱.۵)، توزیع احتمال یکنواخت‌تر می‌شود و کاراکترهای کم‌احتمال‌تر هم شانس بیشتری برای انتخاب دارند؛ متن حاصل خلاقانه‌تر و متنوع‌تر است اما ممکن است دستوری نادرست‌تری داشته باشد. این همان مفهوم تعادل میان «کاوش» و «بهره‌برداری» در یادگیری تقویتی است.

---

# ❓ سؤالات تستی

### سؤال ۱

در پروژه‌ی پیش‌بینی سهام، چرا از HuberLoss به‌جای MSELoss استفاده شد؟

الف) چون HuberLoss سریع‌تر محاسبه می‌شود

ب) چون داده‌های مالی معمولاً مقادیر پرت (Outlier) دارند و HuberLoss نسبت به MSE مقاوم‌تر است، زیرا برای خطاهای بزرگ از نُرم خطی به‌جای مربعی استفاده می‌کند

ج) چون HuberLoss برای سری زمانی طراحی شده

د) چون MSELoss در PyTorch برای چندین ویژگی کار نمی‌کند

### سؤال ۲

در تولید متن با مدل کاراکتری، مفهوم Perplexity چه چیزی را اندازه می‌گیرد؟

الف) تعداد کاراکترهای تولیدشده در هر ثانیه

ب) میانگین تعداد کاراکترهایی که مدل در هر مرحله باید از بین آن‌ها انتخاب کند؛ مقدار کمتر یعنی مدل با اطمینان بیشتری کاراکتر بعدی را پیش‌بینی می‌کند

ج) اندازه‌ی واژگان مدل

د) سرعت آموزش مدل

---

## ✅ پاسخ‌ها

**پاسخ سؤال ۱: گزینه ب**

داده‌های مالی مثل قیمت سهام اغلب دارای جهش‌های ناگهانی و مقادیر پرت هستند. MSE این مقادیر را با توان دو جریمه می‌کند که باعث می‌شود مدل بیش از حد برای کاهش خطا روی این نقاط تلاش کند و از بقیه‌ی الگوها غافل شود. HuberLoss ترکیبی از MSE (برای خطاهای کوچک) و MAE (برای خطاهای بزرگ) است که این تعادل را برقرار می‌کند.

**پاسخ سؤال ۲: گزینه ب**

Perplexity که با نمای خطا محاسبه می‌شود (exp(loss))، نشان می‌دهد که مدل به‌طور میانگین بین چند گزینه مردد است. Perplexity برابر ۱ یعنی مدل با قطعیت کامل پیش‌بینی می‌کند (ایده‌آل اما غیرواقعی)، و Perplexity برابر اندازه‌ی واژگان یعنی مدل کاملاً تصادفی عمل می‌کند. در مدل‌های خوب، Perplexity باید خیلی کمتر از اندازه‌ی واژگان باشد.

---

# 📝 خلاصه فصل

در این فصل دو پروژه‌ی کامل و متفاوت با معماری‌های بازگشتی پیاده‌سازی کردیم. در پروژه‌ی اول، یک مدل LSTM چندویژگی با Attention Layer برای پیش‌بینی قیمت سهام ساختیم و یاد گرفتیم چگونه چندین ویژگی همزمان را نرمال‌سازی و به مدل بدهیم. در پروژه‌ی دوم، یک مدل GRU کاراکتری برای تولید متن پیاده‌سازی کردیم و با مفهوم temperature برای کنترل خلاقیت مدل آشنا شدیم. این دو پروژه نشان می‌دهند که همان خانواده‌ی معماری‌های بازگشتی می‌توانند برای طیف وسیعی از مسائل ترتیبی از پیش‌بینی مالی تا تولید محتوای خلاقانه استفاده شوند.

---

# 🎯 تمرین‌های پایان فصل

۱. در پروژه‌ی سهام، Attention Mechanism را حذف کنید و مدل ساده‌تر را با آن مقایسه کنید.

۲. پروژه‌ی تولید متن را با یک متن فارسی آموزش دهید و نتایج را بررسی کنید.

۳. در پروژه‌ی سهام، به‌جای پیش‌بینی یک روز آینده، سه روز آینده را همزمان پیش‌بینی کنید.

۴. یک مدل زبانی در سطح کلمه (Word-level) به‌جای کاراکتر بسازید و مقایسه کنید.

۵. پروژه‌ی تولید متن را با دمای مختلف از ۰.۲ تا ۲.۰ آزمایش کنید و تأثیر آن را روی کیفیت متن تولیدشده بررسی کنید.

---

با پایان این فصل، بخش سوم کتاب که به داده‌های ترتیبی و شبکه‌های بازگشتی اختصاص داشت به پایان رسید. در **بخش چهارم** که از فصل ۱۹ آغاز می‌شود، وارد دنیای مکانیزم توجه و ترنسفورمرها می‌شویم؛ معماری‌هایی که انقلابی در پردازش زبان طبیعی و فراتر از آن ایجاد کردند.
