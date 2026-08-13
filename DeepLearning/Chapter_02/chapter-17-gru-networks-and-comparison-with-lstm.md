# فصل ۱۷ ⚙️ شبکه‌ی GRU و مقایسه‌ی معماری‌های بازگشتی

## 🎯 اهداف فصل

در فصل قبل با LSTM آشنا شدیم که با سه دروازه و دو حالت داخلی (Cell State و Hidden State) مشکل گرادیان محوشونده را حل می‌کند. در سال ۲۰۱۴، یانگ‌هوا چو و همکارانش معماری ساده‌تری به نام GRU (Gated Recurrent Unit) معرفی کردند که با دو دروازه به‌جای سه دروازه، و یک حالت داخلی به‌جای دو حالت، بسیاری از قابلیت‌های LSTM را با محاسبات کمتر فراهم می‌کند. در پایان این فصل خواهید توانست فرمول هر دو دروازه‌ی GRU را با مثال عددی محاسبه کنید، تفاوت‌های ساختاری GRU و LSTM را توضیح دهید، بدانید در چه موقعیت‌هایی کدام معماری انتخاب بهتری است، و یک مقایسه‌ی عملی کامل بین RNN، LSTM و GRU روی دیتاست واقعی انجام دهید.

---

# 🔄 ایده‌ی اصلی GRU: سادگی بدون از دست دادن قدرت

GRU با این پرسش شروع می‌کند: آیا می‌توانیم بدون Cell State جداگانه و با فقط دو دروازه، همان عملکرد LSTM را داشته باشیم؟ پاسخ در بسیاری از مسائل عملی مثبت است. GRU دو کار اصلی LSTM (فراموش کردن اطلاعات قدیمی و افزودن اطلاعات جدید) را در یک دروازه ادغام می‌کند و Cell State را با Hidden State یکی می‌کند. نتیجه یک معماری سبک‌تر است که اغلب سریع‌تر آموزش می‌گیرد و به داده‌ی کمتری برای همگرایی نیاز دارد.

---

# 📐 فرمول‌های GRU با توضیح کامل

GRU دو دروازه دارد: دروازه‌ی بازنشانی (Reset Gate) و دروازه‌ی به‌روزرسانی (Update Gate).

**دروازه‌ی بازنشانی (r_t)** کنترل می‌کند که چقدر از حالت پنهان قبلی باید در محاسبه‌ی حالت پنهان کاندیدا نقش داشته باشد. وقتی r_t نزدیک صفر باشد، گذشته فراموش می‌شود و حالت جدید فقط بر اساس ورودی فعلی ساخته می‌شود:

```
r_t = σ( W_r × [h_{t-1}, x_t] + b_r )
```

**دروازه‌ی به‌روزرسانی (z_t)** تعیین می‌کند چه مقدار از حالت پنهان قبلی حفظ شود و چه مقدار با اطلاعات جدید جایگزین شود. این دروازه در واقع وظایف هر دوی Forget Gate و Input Gate در LSTM را یک‌جا انجام می‌دهد:

```
z_t = σ( W_z × [h_{t-1}, x_t] + b_z )
```

سپس با استفاده از Reset Gate، حالت پنهان کاندیدا (h̃_t) محاسبه می‌شود:

```
h̃_t = tanh( W_h × [r_t ⊙ h_{t-1}, x_t] + b_h )
```

و در نهایت حالت پنهان جدید از ترکیب حالت قبلی و کاندیدای جدید ساخته می‌شود:

```
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

این فرمول آخر قلب GRU است. وقتی z_t نزدیک یک باشد، h_t تقریباً برابر h̃_t می‌شود (اطلاعات جدید غالب هستند). وقتی z_t نزدیک صفر باشد، h_t تقریباً برابر h_{t-1} می‌شود (حافظه‌ی قدیمی حفظ می‌شود). این مکانیزم از نظر انتقال گرادیان شبیه Skip Connection در ResNet عمل می‌کند.

📷 [تصویر اینجا قرار گیرد: نمودار مقایسه‌ی ساختار سلول LSTM (با سه دروازه و Cell State جداگانه) در برابر GRU (با دو دروازه و یک حالت داخلی) کنار هم]

---

# 🔢 مثال عددی کامل: یک گام GRU

همانند فصل قبل، یک گام کامل را با اعداد ساده محاسبه می‌کنیم. ورودی x_t = 0.8، حالت پنهان قبلی h_{t-1} = 0.3، و برای سادگی تمام وزن‌ها و بایاس‌ها را اسکالر فرض می‌کنیم با وزن برابر ۱ و بایاس برابر ۰:

محاسبه‌ی دروازه‌ها:
```
z_input = 1×0.3 + 1×0.8 = 1.1
r_input = 1×0.3 + 1×0.8 = 1.1

z_t = σ(1.1) = 1/(1+e^{-1.1}) ≈ 0.750
r_t = σ(1.1) ≈ 0.750
```

محاسبه‌ی حالت کاندیدا:
```
candidate_input = r_t × h_{t-1} + x_t
candidate_input = 0.750 × 0.3 + 0.8 = 0.225 + 0.8 = 1.025

h̃_t = tanh(1.025) ≈ 0.770
```

محاسبه‌ی حالت پنهان جدید:
```
h_t = (1 - z_t) × h_{t-1} + z_t × h̃_t
h_t = (1 - 0.750) × 0.3 + 0.750 × 0.770
h_t = 0.250 × 0.3 + 0.750 × 0.770
h_t = 0.075 + 0.578
h_t ≈ 0.653
```

در این گام، حالت پنهان از ۰.۳ به ۰.۶۵۳ رسید. چون z_t برابر ۰.۷۵ بود، ۷۵ درصد وزن به اطلاعات جدید و ۲۵ درصد به حافظه‌ی قدیمی داده شد. اگر z_t نزدیک صفر بود، h_t تقریباً همان ۰.۳ اولیه باقی می‌ماند که یعنی «حافظه را حفظ کن».

---

# ⚖️ مقایسه‌ی LSTM و GRU

| ویژگی | LSTM | GRU |
|---|---|---|
| تعداد دروازه | سه (Forget، Input، Output) | دو (Reset، Update) |
| حالت‌های داخلی | دو (Cell State + Hidden State) | یک (فقط Hidden State) |
| تعداد پارامتر | بیشتر (~۴× hidden_size²) | کمتر (~۳× hidden_size²) |
| سرعت آموزش | کندتر | سریع‌تر |
| عملکرد روی داده‌ی کم | معمولاً بهتر | گاهی بهتر |
| عملکرد روی توالی‌های خیلی بلند | اندکی بهتر | قابل‌مقایسه |
| پیچیدگی پیاده‌سازی | بیشتر | کمتر |

به‌طور کلی، در پروژه‌های صنعتی و وقتی منابع محاسباتی محدود است، GRU انتخاب بهتری است. وقتی داده‌ی زیاد داریم و دقت نهایی اهمیت بیشتری دارد، LSTM ممکن است عملکرد کمی بهتری داشته باشد. در عمل، توصیه‌ی رایج این است که ابتدا با GRU شروع کنید و اگر نتایج کافی نبود، به LSTM تغییر دهید.

---

# 💻 پروژه‌ی عملی فصل: مقایسه‌ی کامل RNN، LSTM و GRU

در این پروژه هر سه معماری را با شرایط یکسان روی دیتاست واقعی پیش‌بینی فروش (Air Passengers Dataset) آموزش می‌دهیم و نتایج را از نظر دقت، سرعت و تعداد پارامتر مقایسه می‌کنیم. این دیتاست شامل تعداد ماهانه‌ی مسافران هوایی بین سال‌های ۱۹۴۹ تا ۱۹۶۰ است و یکی از کلاسیک‌ترین دیتاست‌های سری زمانی است.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from torch.utils.data import Dataset, DataLoader
import time

# ---- بارگذاری دیتاست Air Passengers ----
# داده‌ی معروف تعداد مسافر هوایی ماهانه
air_passengers = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432
]

data = np.array(air_passengers, dtype=np.float32).reshape(-1, 1)

# نرمال‌سازی
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data).flatten()

# ---- ساخت دیتاست توالی ----
class TimeSeriesDataset(Dataset):

    def __init__(self, data, seq_len):
        self.X = []
        self.y = []
        for i in range(len(data) - seq_len):
            self.X.append(torch.tensor(
                data[i:i+seq_len], dtype=torch.float32
            ).unsqueeze(1))
            self.y.append(torch.tensor(
                data[i+seq_len], dtype=torch.float32
            ))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

seq_len = 12  # ۱۲ ماه قبل برای پیش‌بینی ماه بعد
n_train = int(len(data_scaled) * 0.8)

train_dataset = TimeSeriesDataset(data_scaled[:n_train + seq_len], seq_len)
test_dataset = TimeSeriesDataset(data_scaled[n_train:], seq_len)

trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"نمونه‌های آموزش: {len(train_dataset)}")
print(f"نمونه‌های آزمون: {len(test_dataset)}")

# ---- تعریف مدل عمومی برای هر سه معماری ----
class SequenceModel(nn.Module):

    def __init__(self, cell_type='lstm', hidden_size=64, num_layers=2):
        super().__init__()
        self.cell_type = cell_type

        if cell_type == 'rnn':
            self.rnn = nn.RNN(
                1, hidden_size, num_layers,
                batch_first=True, dropout=0.2
            )
        elif cell_type == 'lstm':
            self.rnn = nn.LSTM(
                1, hidden_size, num_layers,
                batch_first=True, dropout=0.2
            )
        elif cell_type == 'gru':
            self.rnn = nn.GRU(
                1, hidden_size, num_layers,
                batch_first=True, dropout=0.2
            )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        if self.cell_type == 'lstm':
            out, (h, c) = self.rnn(x)
        else:
            out, h = self.rnn(x)

        # استفاده از آخرین حالت پنهان
        last_out = out[:, -1, :]
        return self.fc(last_out).squeeze(1)


def train_and_evaluate(cell_type, epochs=100):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SequenceModel(cell_type=cell_type).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0
        for x_batch, y_batch in trainloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in testloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                val_loss += criterion(preds, y_batch).item()

        val_loss /= len(testloader)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    training_time = time.time() - start_time

    # ارزیابی نهایی با MAE در مقیاس اصلی
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in testloader:
            x_batch = x_batch.to(device)
            preds = model(x_batch)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.numpy())

    preds_orig = scaler.inverse_transform(
        np.array(all_preds).reshape(-1, 1)
    ).flatten()
    targets_orig = scaler.inverse_transform(
        np.array(all_targets).reshape(-1, 1)
    ).flatten()

    mae = mean_absolute_error(targets_orig, preds_orig)

    return {
        'cell_type': cell_type.upper(),
        'params': n_params,
        'mae': mae,
        'training_time': training_time
    }


print("\n=== مقایسه‌ی RNN، LSTM و GRU روی Air Passengers Dataset ===\n")
results = []
for cell_type in ['rnn', 'lstm', 'gru']:
    print(f"در حال آموزش {cell_type.upper()}...")
    result = train_and_evaluate(cell_type)
    results.append(result)

print(f"\n{'='*65}")
print(f"{'معماری':<12} {'پارامتر':<15} {'MAE (مسافر)':<20} {'زمان (ثانیه)'}")
print(f"{'-'*65}")
for r in results:
    print(f"{r['cell_type']:<12} {r['params']:<15,} {r['mae']:<20.2f} {r['training_time']:.1f}")
print(f"{'='*65}")

# نمایش برنده در هر دسته
best_acc = min(results, key=lambda x: x['mae'])
fastest = min(results, key=lambda x: x['training_time'])
smallest = min(results, key=lambda x: x['params'])

print(f"\nدقیق‌ترین مدل: {best_acc['cell_type']} با MAE برابر {best_acc['mae']:.2f}")
print(f"سریع‌ترین آموزش: {fastest['cell_type']} با {fastest['training_time']:.1f} ثانیه")
print(f"کم‌پارامترترین مدل: {smallest['cell_type']} با {smallest['params']:,} پارامتر")
```

نتیجه‌ی این مقایسه معمولاً نشان می‌دهد که GRU و LSTM عملکرد بسیار نزدیکی از نظر MAE دارند، اما GRU با پارامترهای کمتر و زمان آموزش کوتاه‌تر به این نتیجه می‌رسد. RNN ساده معمولاً MAE بالاتری دارد، به‌خصوص در این دیتاست که الگوهای فصلی ۱۲ ماهه دارد و نیاز به یادگیری وابستگی‌های طولانی دارد. این آزمایش عملی دقیقاً همان تئوری‌ای را که در فصل‌های ۱۴ و ۱۵ بررسی کردیم، در عمل نشان می‌دهد.

---

# ❓ سؤالات تستی

### سؤال ۱

در فرمول GRU، h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t، وقتی z_t برابر ۱ باشد چه اتفاقی می‌افتد؟

الف) h_t برابر h_{t-1} می‌شود و حافظه‌ی قبلی کاملاً حفظ می‌شود

ب) h_t برابر h̃_t می‌شود یعنی حالت پنهان کاملاً با اطلاعات جدید بازنویسی می‌شود

ج) h_t برابر صفر می‌شود

د) h_t برابر میانگین h_{t-1} و h̃_t می‌شود

### سؤال ۲

کدام‌یک از موارد زیر یک دلیل عملی برای انتخاب GRU به‌جای LSTM است؟

الف) GRU همیشه دقت بالاتری نسبت به LSTM دارد

ب) GRU با پارامترهای کمتر و زمان آموزش کوتاه‌تر، در بسیاری از مسائل عملی عملکردی مشابه LSTM دارد

ج) GRU از LSTM قدیمی‌تر است

د) GRU برای داده‌های تصویری مناسب‌تر است

---

## ✅ پاسخ‌ها

**پاسخ سؤال ۱: گزینه ب**

وقتی z_t = ۱، عبارت (1-z_t) = ۰ می‌شود و جمله‌ی اول h_{t-1} حذف می‌شود. در نتیجه h_t = z_t ⊙ h̃_t = ۱ × h̃_t = h̃_t. این یعنی حالت پنهان کاملاً با اطلاعات جدید جایگزین شده و گذشته فراموش شده است. حالت مقابل (z_t = ۰) باعث می‌شود h_t = h_{t-1} یعنی حافظه کاملاً حفظ شود.

**پاسخ سؤال ۲: گزینه ب**

GRU با یک سوم پارامتر کمتر نسبت به LSTM (چون دو دروازه دارد نه سه دروازه، و Cell State جداگانه ندارد) در بسیاری از مسائل عملی عملکرد مشابهی دارد. این به معنای آموزش سریع‌تر، مصرف حافظه‌ی کمتر و نیاز به داده‌ی کمتر برای همگرایی است که در پروژه‌های صنعتی با منابع محدود اهمیت زیادی دارد.

---

# 📝 خلاصه فصل

در این فصل معماری GRU را به‌عنوان نسخه‌ی ساده‌شده و کارآمد LSTM بررسی کردیم. دو دروازه‌ی Reset و Update را با فرمول کامل و مثال عددی گام‌به‌گام محاسبه کردیم. دیدیم که دروازه‌ی Update در GRU همان کاری را می‌کند که Forget Gate و Input Gate در LSTM با هم انجام می‌دهند، و حذف Cell State جداگانه ساختار را ساده‌تر کرده بدون اینکه قدرت یادگیری وابستگی‌های طولانی را از بین ببرد. در مقایسه‌ی جدول‌بندی‌شده و پروژه‌ی عملی هم دیدیم که GRU در اکثر مسائل عملی پارامتر کمتر و سرعت آموزش بیشتری دارد در حالی که دقت مشابهی ارائه می‌دهد.

---

# 🎯 تمرین‌های پایان فصل

۱. یک گام کامل GRU را با مقادیر متفاوت از مثال این فصل روی کاغذ محاسبه کنید و نتیجه را با PyTorch تأیید کنید.

۲. در پروژه‌ی این فصل، hidden_size را از ۶۴ به ۱۲۸ دوبرابر کنید و تأثیر آن بر MAE و زمان آموزش را برای هر سه معماری مقایسه کنید.

۳. یک Bidirectional GRU پیاده‌سازی کنید و نتیجه را با GRU یک‌طرفه مقایسه کنید.

۴. به‌جای استفاده از آخرین گام زمانی، از Attention Mechanism ساده‌ای روی تمام خروجی‌های GRU برای وزن‌دهی و ترکیب آن‌ها استفاده کنید.

۵. دیتاست پیش‌بینی دمای شهر دهلی از فصل ۱۴ را دوباره با LSTM و GRU آموزش دهید و با نتیجه‌ی RNN ساده مقایسه کنید.

---

در فصل ۱۸ به **پروژه‌ی جامع بخش سوم** می‌رسیم؛ جایی که LSTM و GRU را روی دو مسئله‌ی واقعی کاملاً متفاوت (پیش‌بینی سری زمانی مالی و تولید متن) با هم ترکیب می‌کنیم تا درک عمیقی از کاربردهای عملی این معماری‌ها به دست آوریم.
