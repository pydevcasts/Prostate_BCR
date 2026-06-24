# فصل ۱۵ ⚠️ مشکل گرادیان محوشونده و انفجاری در شبکه‌های بازگشتی

## 🎯 اهداف فصل

در فصل قبل با RNN آشنا شدیم و دیدیم که در یادگیری وابستگی‌های طولانی ضعیف عمل می‌کند. در این فصل به دلیل ریاضی دقیق این ضعف می‌پردازیم. درک عمیق این مشکل برای دو دلیل مهم است: اول، بدون شناخت دقیق مشکل، نمی‌توان راه‌حل آن یعنی LSTM و GRU را که در فصل‌های بعدی می‌آیند، واقعاً درک کرد؛ دوم، این مشکل به اشکال مختلف در شبکه‌های عمیق هم رخ می‌دهد و شناخت ریشه‌ی آن برای هر مهندس یادگیری عمیق ضروری است. در پایان این فصل خواهید توانست با اشتقاق ریاضی نشان دهید که چرا گرادیان در طول توالی‌های طولانی محو یا منفجر می‌شود، تأثیر این مشکل را با آزمایش عملی مستقیماً مشاهده کنید، و تکنیک‌های کاهش اثر این مشکل مثل Gradient Clipping را پیاده‌سازی کنید.

---

# 🔙 پس‌انتشار خطا از طریق زمان (BPTT)

برای درک مشکل گرادیان محوشونده، ابتدا باید بفهمیم که الگوریتم پس‌انتشار خطا چگونه در RNN عمل می‌کند. در یک شبکه‌ی معمولی، پس‌انتشار خطا از لایه‌ی آخر به لایه‌ی اول در عمق شبکه اعمال می‌شود. در RNN، یک بُعد اضافی هم داریم: زمان. الگوریتم پس‌انتشار خطا در RNN باید گرادیان را نه فقط در عمق لایه‌ها، بلکه در طول گام‌های زمانی هم منتقل کند. به همین دلیل این الگوریتم را Back-Propagation Through Time یا BPTT می‌نامند.

تصور کنید یک توالی صد عنصری داریم. در BPTT، گرادیان خطا از گام زمانی صدم شروع می‌کند و باید گام‌به‌گام به سمت گام اول برگردد. در هر گام زمانی، گرادیان با مشتق تابع فعال‌سازی و ماتریس وزن W_hh ضرب می‌شود. اگر این شبکه صد گام زمانی داشته باشد، گرادیان باید صد بار این ضرب را تجربه کند. همین تکرار مداوم ضرب است که مشکل اصلی را ایجاد می‌کند.

---

# 📐 اشتقاق ریاضی: چرا گرادیان محو می‌شود؟

برای اشتقاق دقیق، فرمول به‌روزرسانی حالت پنهان را به خاطر می‌آوریم:

```
h_t = tanh( W_hh * h_{t-1} + W_xh * x_t + b )
```

برای محاسبه‌ی گرادیان خطا L نسبت به h_1 (حالت پنهان گام اول) در یک توالی T گامه، باید قانون زنجیره‌ای را از گام T تا گام ۱ اعمال کنیم:

```
∂L/∂h_1 = ∂L/∂h_T × ∏(t=2 to T) [ ∂h_t/∂h_{t-1} ]
```

مشتق h_t نسبت به h_{t-1} برابر است با:

```
∂h_t/∂h_{t-1} = diag(1 - h_t²) × W_hh
```

در این فرمول، عبارت diag(1 - h_t²) قطری‌شده‌ی مشتق تابع Tanh است. مقدار مشتق Tanh در هر نقطه بین صفر و یک قرار دارد. پس نُرم این حاصل‌ضرب به شکل زیر کران‌دار می‌شود:

```
‖∂h_t/∂h_{t-1}‖ ≤ ‖W_hh‖ × max(1 - h_t²)
```

حال اگر بزرگ‌ترین مقدار تکین ماتریس W_hh را λ_max بنامیم، گرادیان کل به شکل زیر رفتار می‌کند:

```
‖∂L/∂h_1‖ ≈ ‖∂L/∂h_T‖ × λ_max^(T-1)
```

این فرمول تمام ماجرا را روشن می‌کند. اگر λ_max کوچک‌تر از یک باشد، λ_max^(T-1) با بزرگ‌شدن T به سمت صفر میل می‌کند؛ این همان گرادیان محوشونده است. اما اگر λ_max بزرگ‌تر از یک باشد، همین عبارت به سمت بی‌نهایت می‌رود؛ این گرادیان انفجاری است.

برای یک مثال عددی ساده: فرض کنید λ_max برابر نه دهم باشد و توالی پنجاه گام داشته باشیم. گرادیان با ضریب نه دهم به توان چهل و نه ضرب می‌شود که برابر با تقریباً پنج هزارم (۰.۰۰۵) است. یعنی گرادیانی که در گام پنجاهم یک بود، تا وقتی به گام اول برسد به پنج هزارم کاهش یافته است. این مقدار آن‌قدر کوچک است که عملاً هیچ به‌روزرسانی در وزن‌های مرتبط با گام‌های اولیه اتفاق نمی‌افتد.

📷 [تصویر اینجا قرار گیرد: نمودار دو گراف کنار هم؛ سمت چپ نشان می‌دهد که λ < 1 باعث کاهش نمایی گرادیان در طول زمان می‌شود، سمت راست نشان می‌دهد که λ > 1 باعث افزایش نمایی گرادیان می‌شود]

---

# 🔬 آزمایش عملی: مشاهده‌ی مستقیم محوشدگی گرادیان

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def measure_gradient_flow(seq_lengths, hidden_size=32):
    """
    برای هر طول توالی، یک RNN ساده می‌سازیم و
    اندازه‌ی گرادیان در گام‌های مختلف زمانی را اندازه می‌گیریم
    """

    results = {}

    for seq_len in seq_lengths:

        # ساخت یک RNN ساده
        rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True
        )
        fc = nn.Linear(hidden_size, 1)

        # ورودی ساختگی با نیاز به گرادیان
        x = torch.randn(1, seq_len, 1, requires_grad=True)
        h0 = torch.zeros(1, 1, hidden_size)

        # انتشار پیشرو
        out, _ = rnn(x, h0)
        pred = fc(out[:, -1, :])
        target = torch.tensor([[1.0]])
        loss = nn.MSELoss()(pred, target)

        # انتشار پسرو
        loss.backward()

        # ثبت نُرم گرادیان در طول زمان
        grad_norms = []
        # گرادیان x.grad شکل (1, seq_len, 1) دارد
        for t in range(seq_len):
            grad_norm = x.grad[0, t, 0].abs().item()
            grad_norms.append(grad_norm)

        results[seq_len] = grad_norms

    return results


seq_lengths = [10, 30, 50, 100]
results = measure_gradient_flow(seq_lengths)

print("اندازه‌ی گرادیان در گام‌های مختلف زمانی:")
print(f"{'طول توالی':<15} {'گام اول':<15} {'گام میانی':<15} {'گام آخر':<15}")
print("-" * 60)
for seq_len, grads in results.items():
    first = grads[0]
    mid = grads[seq_len // 2]
    last = grads[-1]
    print(f"{seq_len:<15} {first:<15.2e} {mid:<15.2e} {last:<15.2e}")

print("\nنتیجه: با افزایش طول توالی، گرادیان در گام‌های اولیه")
print("به شدت کاهش می‌یابد که همان مشکل گرادیان محوشونده است.")
```

---

# 🔧 راه‌حل‌های موقت: Gradient Clipping و مقداردهی اولیه

در حالی که راه‌حل اصلی مشکل گرادیان محوشونده معماری‌های LSTM و GRU هستند که در فصل‌های بعدی می‌آیند، چند تکنیک کوتاه‌مدت برای کاهش اثر این مشکل وجود دارد. مهم‌ترین آن‌ها Gradient Clipping است که برای مشکل گرادیان انفجاری (نه محوشونده) بسیار مؤثر است.

Gradient Clipping نُرم گرادیان کل را قبل از به‌روزرسانی وزن‌ها محدود می‌کند. اگر نُرم گرادیان از یک حداکثر مشخص (مثلاً یک یا پنج) بزرگ‌تر شد، تمام گرادیان‌ها به‌صورت متناسب کوچک‌تر می‌شوند تا نُرم کلی برابر آن حداکثر شود:

```
اگر ‖g‖ > max_norm:
    g = g × (max_norm / ‖g‖)
```

تکنیک دیگر، مقداردهی اولیه‌ی هوشمند وزن‌هاست. یکی از روش‌های مؤثر برای RNN، مقداردهی W_hh به‌عنوان یک ماتریس متعامد (Orthogonal Matrix) است؛ این ماتریس‌ها ویژگی خاصی دارند که نُرم بردارها را در ضرب ماتریسی حفظ می‌کنند و از گسترش یا کوچک‌شدن آن‌ها در طول زمان جلوگیری می‌کنند.

```python
# مقایسه‌ی تأثیر Gradient Clipping روی پایداری آموزش RNN
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleRNN_Test(nn.Module):

    def __init__(self, hidden_size=64):
        super().__init__()
        self.rnn = nn.RNN(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze()


def train_with_clipping(use_clipping, seq_len=50, epochs=30):

    torch.manual_seed(42)
    model = SimpleRNN_Test()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # داده‌ی ساختگی: توالی اعداد تصادفی
    X = torch.randn(100, seq_len, 1)
    y = X.mean(dim=1).squeeze()

    losses = []
    max_grads = []

    for epoch in range(epochs):

        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()

        # اندازه‌گیری نُرم گرادیان قبل از Clipping
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        max_grads.append(total_norm)

        # اعمال یا عدم اعمال Gradient Clipping
        if use_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        losses.append(loss.item())

    return losses, max_grads


print("آزمایش: تأثیر Gradient Clipping بر پایداری آموزش\n")

losses_no_clip, grads_no_clip = train_with_clipping(use_clipping=False)
losses_with_clip, grads_with_clip = train_with_clipping(use_clipping=True)

print(f"بدون Gradient Clipping:")
print(f"  حداکثر نُرم گرادیان: {max(grads_no_clip):.2f}")
print(f"  خطای اولیه: {losses_no_clip[0]:.4f}")
print(f"  خطای نهایی: {losses_no_clip[-1]:.4f}")
print(f"  آیا خطا NaN شد؟ {any(np.isnan(l) for l in losses_no_clip)}")

print(f"\nبا Gradient Clipping (max_norm=1.0):")
print(f"  حداکثر نُرم گرادیان: {max(grads_with_clip):.2f}")
print(f"  خطای اولیه: {losses_with_clip[0]:.4f}")
print(f"  خطای نهایی: {losses_with_clip[-1]:.4f}")
print(f"  آیا خطا NaN شد؟ {any(np.isnan(l) for l in losses_with_clip)}")
```

---

# 🔬 مقایسه‌ی تجربی: RNN ساده روی وابستگی‌های کوتاه و بلند

```python
def test_long_term_dependency(dependency_length):
    """
    آزمایش توانایی RNN ساده در یادگیری وابستگی‌های با فاصله‌های مختلف.
    مسئله: پیش‌بینی اینکه اولین عنصر توالی مثبت بود یا منفی
    (وابستگی dependency_length گام دور)
    """

    torch.manual_seed(42)
    seq_len = dependency_length + 5

    # ساخت داده
    # اولین عنصر هر توالی برچسب را تعیین می‌کند
    n_samples = 500
    X = torch.randn(n_samples, seq_len, 1) * 0.1
    # اولین ورودی هر توالی مثبت یا منفی است
    first_vals = (torch.randint(0, 2, (n_samples,)) * 2 - 1).float()
    X[:, 0, 0] = first_vals * 2.0
    # هدف: پیش‌بینی علامت اولین عنصر بعد از dependency_length گام
    y = (first_vals > 0).float()

    # تقسیم داده
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]

    model = nn.Sequential(
        nn.RNN(1, 32, batch_first=True),
    )

    # مدل ساده برای آزمایش
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.RNN(1, 32, batch_first=True)
            self.fc = nn.Linear(32, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.sigmoid(self.fc(out[:, -1, :])).squeeze()

    model = TestModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_preds = (model(X_test) > 0.5).float()
        acc = (test_preds == y_test).float().mean().item() * 100

    return acc


print("آزمایش توانایی RNN در یادگیری وابستگی‌های با فاصله‌های مختلف:\n")
print(f"{'فاصله وابستگی':<20} {'دقت RNN ساده':<20} {'مقایسه با تصادفی'}")
print("-" * 60)

for dep_len in [5, 10, 20, 50]:
    acc = test_long_term_dependency(dep_len)
    vs_random = "بهتر" if acc > 60 else "مشابه یا بدتر"
    print(f"{dep_len:<20} {acc:.1f}%{'':<14} {vs_random}")

print("\nنتیجه: با افزایش فاصله‌ی وابستگی، دقت RNN ساده کاهش می‌یابد")
print("و در وابستگی‌های خیلی دور، به دقت تصادفی (۵۰٪) نزدیک می‌شود.")
```

---

# ❓ سؤالات تستی

### سؤال ۱

در فرمول انتشار گرادیان در RNN، مقدار λ_max^(T-1) چه نقشی دارد؟

الف) تعداد لایه‌های شبکه را تعیین می‌کند

ب) نشان می‌دهد که گرادیان با نمای (T-1) در بزرگ‌ترین مقدار تکین W_hh ضرب می‌شود؛ اگر این مقدار کمتر از یک باشد گرادیان محو می‌شود و اگر بیشتر از یک باشد منفجر می‌شود

ج) اندازه‌ی حالت پنهان را مشخص می‌کند

د) نرخ یادگیری بهینه را محاسبه می‌کند

### سؤال ۲

Gradient Clipping برای حل کدام‌یک از دو مشکل مؤثرتر است: گرادیان محوشونده یا گرادیان انفجاری؟

الف) گرادیان محوشونده، چون گرادیان‌های کوچک را بزرگ‌تر می‌کند

ب) گرادیان انفجاری، چون گرادیان‌های بسیار بزرگ را محدود می‌کند و از ناپایداری آموزش جلوگیری می‌کند

ج) هر دو به‌طور یکسان

د) هیچ‌کدام، چون Gradient Clipping فقط برای شبکه‌های MLP کاربرد دارد

---

## ✅ پاسخ‌ها

**پاسخ سؤال ۱: گزینه ب**

عبارت λ_max^(T-1) نشان می‌دهد که با هر گام زمانی اضافه، یک ضرب دیگر در λ_max انجام می‌شود. این ضرب تکراری است که باعث رفتار نمایی گرادیان می‌شود؛ نمایی کاهشی اگر λ_max < 1 باشد (محوشدگی) و نمایی افزایشی اگر λ_max > 1 باشد (انفجار). درک این رفتار نمایی، دلیل اصلی جدی بودن این مشکل برای توالی‌های طولانی را روشن می‌کند.

**پاسخ سؤال ۲: گزینه ب**

Gradient Clipping با محدود کردن نُرم گرادیان از رشد بی‌رویه‌ی آن جلوگیری می‌کند و مستقیماً مشکل انفجار را حل می‌کند. اما برای گرادیان محوشونده که مشکل خیلی کوچک بودن گرادیان است، این تکنیک کمکی نمی‌کند؛ راه‌حل آن معماری‌هایی هستند که مسیر مستقیم‌تری برای انتقال گرادیان فراهم می‌کنند، مثل Skip Connection در ResNet یا دروازه‌ها در LSTM.

---

# 📝 خلاصه فصل

در این فصل ریشه‌ی ریاضی مشکل گرادیان محوشونده و انفجاری را در RNN با اشتقاق کامل بررسی کردیم. دیدیم که رفتار نمایی λ_max^(T-1) تعیین می‌کند که گرادیان در طول توالی چگونه رفتار کند: اگر بزرگ‌ترین مقدار تکین ماتریس وزن کمتر از یک باشد گرادیان محو می‌شود، و اگر بیشتر از یک باشد گرادیان منفجر می‌شود. این مشکل را با آزمایش‌های عملی مستقیماً مشاهده کردیم و نشان دادیم که با افزایش فاصله‌ی وابستگی، دقت RNN ساده به سمت حدس تصادفی کاهش می‌یابد. Gradient Clipping را به‌عنوان یک راه‌حل موقت برای مشکل انفجار بررسی کردیم. در فصل بعد، راه‌حل اصلی یعنی معماری LSTM را بررسی می‌کنیم که با مکانیزم دروازه‌های هوشمند، جریان گرادیان را در طول توالی‌های طولانی کنترل می‌کند.

---

# 🎯 تمرین‌های پایان فصل

۱. با قانون زنجیره‌ای، فرمول ∂h_t/∂h_{t-1} = diag(1 - h_t²) × W_hh را به‌صورت دستی اشتقاق کنید.

۲. مقدار λ_max^(T-1) را برای λ_max = ۰.۸ و T = ۲۰، ۵۰، ۱۰۰ محاسبه و مقایسه کنید.

۳. در آزمایش Gradient Clipping، مقدار max_norm را از ۱.۰ به ۰.۱ و ۵.۰ تغییر دهید و تأثیر آن بر سرعت همگرایی را بررسی کنید.

۴. مقداردهی اولیه‌ی Orthogonal را برای W_hh در کد آزمایش وابستگی‌های طولانی پیاده‌سازی کنید (`nn.init.orthogonal_`) و ببینید آیا نتایج بهبود می‌یابند.

۵. آزمایش وابستگی‌های طولانی را با دو لایه‌ی RNN (num_layers=2) تکرار کنید و تأثیر عمق بر یادگیری وابستگی‌های دور را بررسی کنید.

---

در فصل ۱۶ به معماری LSTM می‌پردازیم؛ معماری‌ای که با طراحی هوشمندانه‌ی چهار دروازه (Forget، Input، Output و Cell State) راه‌حل اصلی مشکل گرادیان محوشونده را فراهم می‌کند و یادگیری وابستگی‌های طولانی را ممکن می‌سازد.
