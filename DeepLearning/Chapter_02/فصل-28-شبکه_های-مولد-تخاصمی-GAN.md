# فصل ۲۸ 🎭 شبکه‌های مولد تخاصمی (GAN)

## 🎯 اهداف فصل

در فصل قبل با Autoencoder و VAE آشنا شدیم که می‌توانند داده‌ی جدید تولید کنند. در این فصل با یکی از جذاب‌ترین و قدرتمندترین ایده‌های یادگیری عمیق آشنا می‌شویم: شبکه‌های مولد تخاصمی یا GAN که مخفف Generative Adversarial Network است. GAN که در سال ۲۰۱۴ توسط ایان گودفلو معرفی شد، با یک ایده‌ی بازی‌گونه کار می‌کند: دو شبکه‌ی عصبی را در مقابل هم قرار بده تا هر کدام از دیگری بیاموزد. نتیجه مدل‌هایی است که می‌توانند تصاویر، صدا و متن کاملاً واقع‌گرایانه تولید کنند. در پایان این فصل خواهید توانست معماری GAN شامل Generator و Discriminator را با فرمول کامل توضیح دهید، فرآیند آموزش رقابتی و تابع هزینه‌ی GAN را درک کنید، چالش‌های آموزش GAN مثل Mode Collapse را بشناسید، یک GAN کامل را روی داده‌ی واقعی پیاده‌سازی کنید، و انواع پیشرفته‌ی GAN مثل DCGAN و Conditional GAN را بشناسید.

---

# 🎮 ایده‌ی اصلی GAN: بازی دوطرفه‌ی جعل و تشخیص

تصور کنید یک جاعل اسکناس (Generator) می‌خواهد اسکناس‌های تقلبی بسازد که از اسکناس‌های واقعی قابل‌تشخیص نباشند، و یک کارشناس بانک (Discriminator) می‌خواهد اسکناس‌های واقعی را از تقلبی تشخیص دهد. در ابتدا جاعل اسکناس‌های بدی می‌سازد که به‌راحتی کشف می‌شوند. اما با هر شکست، چیز جدیدی یاد می‌گیرد و اسکناس‌های بهتری می‌سازد. در مقابل، کارشناس هم با دیدن اسکناس‌های بهتر، مهارت تشخیص خود را تقویت می‌کند. این رقابت آنقدر ادامه می‌یابد که جاعل به‌قدری خوب می‌شود که کارشناس دیگر نمی‌تواند اسکناس واقعی را از تقلبی تشخیص دهد.

دقیقاً همین فرآیند در GAN رخ می‌دهد. Generator از یک بردار تصادفی z (نویز) یک داده‌ی مصنوعی (مثل تصویر) می‌سازد. Discriminator یک داده می‌گیرد (که می‌تواند واقعی یا مصنوعی باشد) و احتمال واقعی‌بودن آن را برمی‌گرداند. Generator می‌خواهد Discriminator را فریب دهد، و Discriminator می‌خواهد واقعی را از مصنوعی تشخیص دهد.

📷 [تصویر اینجا قرار گیرد: نمودار معماری GAN با Generator در سمت چپ که از نویز z تصویر مصنوعی می‌سازد و Discriminator در سمت راست که هم تصاویر واقعی و هم مصنوعی می‌گیرد و احتمال واقعی‌بودن را برمی‌گرداند]

---

# 📐 فرمول ریاضی GAN

بازی میان Generator (G) و Discriminator (D) را می‌توان به‌صورت یک بازی minimax فرموله کرد. Discriminator می‌خواهد این مقدار را بیشینه کند (چون می‌خواهد واقعی را درست تشخیص دهد و مصنوعی را رد کند) و Generator می‌خواهد آن را کمینه کند (چون می‌خواهد Discriminator را فریب دهد):

```
min_G max_D V(D, G) =
    E[log D(x)]          ← Discriminator روی داده واقعی
  + E[log(1 - D(G(z)))]  ← Discriminator روی داده مصنوعی
```

در این فرمول، x نمونه‌ی واقعی از داده است، z بردار نویز تصادفی است، G(z) تصویر ساخته‌شده توسط Generator است، و D(x) احتمال واقعی‌بودن x است که Discriminator برمی‌گرداند. حالت بهینه‌ی نظری زمانی است که Generator آن‌قدر خوب شود که D(G(z)) برابر نیم شود؛ یعنی Discriminator نتواند بهتر از یک حدس تصادفی عمل کند.

در عمل، برای آموزش Generator از یک تابع هزینه‌ی کمی متفاوت استفاده می‌شود که گرادیان‌های بهتری در ابتدای آموزش دارد:

```
تابع هزینه‌ی Generator: -E[log D(G(z))]
تابع هزینه‌ی Discriminator: -E[log D(x)] - E[log(1 - D(G(z)))]
```

---

# 🔢 مثال عددی: یک گام آموزش GAN

برای درک بهتر فرآیند آموزش، یک گام کامل را با اعداد ساده دنبال می‌کنیم. فرض کنید Discriminator برای یک نمونه‌ی واقعی x احتمال ۰.۸ را برمی‌گرداند (یعنی می‌گوید ۸۰ درصد احتمال دارد واقعی باشد) و برای یک نمونه‌ی مصنوعی G(z) احتمال ۰.۳ را برمی‌گرداند.

خطای Discriminator برابر است با منفی لگاریتم ۰.۸ منهای لگاریتم ۱ منهای ۰.۳، که برابر است با منفی ۰.۲۲ منهای (منفی ۰.۱۵)، یعنی برابر ۰.۳۷. با این گرادیان، Discriminator وزن‌هایش را به‌گونه‌ای به‌روز می‌کند که دفعه‌ی بعد بهتر تشخیص دهد.

سپس نوبت آموزش Generator می‌رسد. خطای Generator برابر است با منفی لگاریتم D(G(z)) که برابر است با منفی لگاریتم ۰.۳، یعنی تقریباً ۱.۲. این عدد بزرگ نشان می‌دهد که Generator هنوز ضعیف است و باید بهبود یابد. با به‌روزرسانی وزن‌های Generator (و ثابت نگه داشتن Discriminator)، Generator یاد می‌گیرد تصاویری بسازد که Discriminator احتمال بالاتری به آن‌ها بدهد.

---

# ⚠️ چالش‌های آموزش GAN

آموزش GAN به دلیل ماهیت رقابتی‌اش از یادگیری یک مدل ساده بسیار دشوارتر است. مهم‌ترین چالش‌ها عبارتند از: نخست Mode Collapse که در آن Generator یاد می‌گیرد فقط چند نوع خروجی (Mode) تولید کند که Discriminator را فریب می‌دهند و تنوع داده‌ی واقعی را نادیده می‌گیرد. دوم عدم تعادل آموزشی که در آن اگر Discriminator خیلی سریع‌تر از Generator یاد بگیرد، گرادیان Generator تقریباً صفر می‌شود و یادگیری متوقف می‌شود. سوم ناپایداری آموزش که در آن خطاها به‌جای کاهش تدریجی نوسان می‌کنند و تصاویر تولیدشده کیفیت ثابتی ندارند.

📷 [تصویر اینجا قرار گیرد: نمودار مقایسه‌ی Mode Collapse (تنها چند نوع خروجی تولید می‌شود) در برابر آموزش موفق (تنوع کامل داده)]

---

# 🏗️ DCGAN: GAN برای تصویر

DCGAN یا Deep Convolutional GAN که در سال ۲۰۱۵ معرفی شد، اولین معماری پایدار GAN برای تصاویر بود. نوآوری‌های اصلی DCGAN عبارتند از: استفاده از لایه‌های کانولوشنی به‌جای لایه‌های تمام‌متصل در هر دو Generator و Discriminator، استفاده از Batch Normalization در اکثر لایه‌ها، استفاده از ReLU در Generator و LeakyReLU در Discriminator، و حذف لایه‌های Pooling و استفاده از Strided Convolution برای کاهش ابعاد.

---

# 💻 پروژه‌ی عملی: DCGAN روی دیتاست MNIST

در این پروژه یک DCGAN کامل برای تولید تصاویر ارقام دست‌نویس روی دیتاست MNIST پیاده‌سازی می‌کنیم.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

# ---- آماده‌سازی داده ----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # نرمال‌سازی به بازه [-1, 1]
])

trainset = torchvision.datasets.MNIST(
    './data', train=True, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

LATENT_DIM = 100  # بُعد بردار نویز ورودی Generator
IMAGE_SIZE = 28
CHANNELS = 1

# ---- تعریف Generator ----
class Generator(nn.Module):
    """
    Generator: از بردار نویز z یک تصویر می‌سازد.
    از Transposed Convolution برای افزایش تدریجی ابعاد استفاده می‌کند.
    """

    def __init__(self, latent_dim=100):
        super().__init__()

        self.network = nn.Sequential(
            # ورودی: (batch, latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # → (batch, 256, 7, 7)

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # → (batch, 128, 14, 14)

            nn.ConvTranspose2d(128, CHANNELS, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # → (batch, 1, 28, 28) با مقادیر در [-1, 1]
        )

    def forward(self, z):
        # z شکل (batch, latent_dim) دارد، باید به (batch, latent_dim, 1, 1) تبدیل شود
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.network(z)


# ---- تعریف Discriminator ----
class Discriminator(nn.Module):
    """
    Discriminator: احتمال واقعی‌بودن یک تصویر را تخمین می‌زند.
    از LeakyReLU برای جلوگیری از مشکل گرادیان صفر استفاده می‌کند.
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            # ورودی: (batch, 1, 28, 28)
            nn.Conv2d(CHANNELS, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 64, 14, 14)

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 128, 7, 7)

            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # → (batch, 1, 1, 1): احتمال واقعی‌بودن
        )

    def forward(self, img):
        return self.network(img).view(-1)


def weights_init(m):
    """مقداردهی اولیه‌ی وزن‌ها بر اساس توصیه‌ی مقاله‌ی DCGAN"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"دستگاه محاسباتی: {device}")

G = Generator(LATENT_DIM).to(device)
D = Discriminator().to(device)

G.apply(weights_init)
D.apply(weights_init)

n_params_G = sum(p.numel() for p in G.parameters())
n_params_D = sum(p.numel() for p in D.parameters())
print(f"پارامترهای Generator: {n_params_G:,}")
print(f"پارامترهای Discriminator: {n_params_D:,}")

# ---- تأیید شکل خروجی ----
z_test = torch.randn(4, LATENT_DIM).to(device)
img_test = G(z_test)
prob_test = D(img_test)
print(f"\nشکل خروجی Generator: {img_test.shape}")
print(f"شکل خروجی Discriminator: {prob_test.shape}")

# ---- تعریف تابع هزینه و بهینه‌سازها ----
criterion = nn.BCELoss()

# بهینه‌ساز جداگانه برای G و D با نرخ یادگیری پایین
optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# یک بردار نویز ثابت برای مشاهده‌ی پیشرفت در طول آموزش
fixed_noise = torch.randn(16, LATENT_DIM).to(device)

# ---- حلقه‌ی آموزش GAN ----
NUM_EPOCHS = 30
d_losses = []
g_losses = []

print("\n=== شروع آموزش DCGAN ===\n")

for epoch in range(NUM_EPOCHS):

    epoch_d_loss = 0.0
    epoch_g_loss = 0.0
    num_batches = 0

    for real_imgs, _ in trainloader:
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)

        # برچسب‌های واقعی (1) و مصنوعی (0) با Label Smoothing
        real_labels = torch.full((batch_size,), 0.9, device=device)  # 0.9 به‌جای 1.0
        fake_labels = torch.zeros(batch_size, device=device)

        # ======== آموزش Discriminator ========
        optimizer_D.zero_grad()

        # خطا روی داده‌ی واقعی
        output_real = D(real_imgs)
        loss_real = criterion(output_real, real_labels)

        # خطا روی داده‌ی مصنوعی
        z = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_imgs = G(z).detach()  # detach تا گرادیان به G نرسد
        output_fake = D(fake_imgs)
        loss_fake = criterion(output_fake, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # ======== آموزش Generator ========
        optimizer_G.zero_grad()

        # Generator می‌خواهد Discriminator بگوید تصاویرش واقعی هستند
        z = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_imgs = G(z)
        output = D(fake_imgs)
        # هدف Generator: D(G(z)) = 1
        loss_G = criterion(output, torch.ones(batch_size, device=device))
        loss_G.backward()
        optimizer_G.step()

        epoch_d_loss += loss_D.item()
        epoch_g_loss += loss_G.item()
        num_batches += 1

    avg_d = epoch_d_loss / num_batches
    avg_g = epoch_g_loss / num_batches
    d_losses.append(avg_d)
    g_losses.append(avg_g)

    if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
        print(f"دوره {epoch+1:3d}/{NUM_EPOCHS} | "
              f"خطای D: {avg_d:.4f} | خطای G: {avg_g:.4f}")

        # ارزیابی کیفیت تصاویر تولیدی
        G.eval()
        with torch.no_grad():
            sample_imgs = G(fixed_noise)
            pixel_range = (sample_imgs.min().item(), sample_imgs.max().item())
            mean_val = sample_imgs.mean().item()
        G.train()
        print(f"         تصاویر نمونه — دامنه: [{pixel_range[0]:.2f}, "
              f"{pixel_range[1]:.2f}] | میانگین: {mean_val:.3f}")


# ---- ارزیابی نهایی و تولید تصاویر ----
G.eval()
D.eval()

print("\n=== ارزیابی نهایی ===")

with torch.no_grad():
    # تولید تصاویر جدید
    z_new = torch.randn(64, LATENT_DIM, device=device)
    generated = G(z_new)
    print(f"تعداد تصاویر تولیدشده: {generated.shape[0]}")
    print(f"شکل هر تصویر: {generated.shape[1:]}")
    print(f"دامنه‌ی مقادیر: [{generated.min().item():.3f}, "
          f"{generated.max().item():.3f}]")

    # بررسی Discriminator روی تصاویر تولیدی
    d_scores = D(generated)
    print(f"\nنمره‌ی میانگین Discriminator برای تصاویر مصنوعی: "
          f"{d_scores.mean().item():.3f}")
    print(f"(نزدیک‌تر به 0.5 = Generator بهتر فریب داده)")

    # بررسی روی تصاویر واقعی
    real_batch, _ = next(iter(trainloader))
    real_scores = D(real_batch[:64].to(device))
    print(f"نمره‌ی میانگین Discriminator برای تصاویر واقعی: "
          f"{real_scores.mean().item():.3f}")
    print(f"(نزدیک‌تر به 0.5 = Discriminator نمی‌تواند تشخیص دهد)")

# ذخیره‌ی مدل‌ها
torch.save(G.state_dict(), 'generator_mnist.pt')
torch.save(D.state_dict(), 'discriminator_mnist.pt')
print("\nمدل‌های Generator و Discriminator ذخیره شدند.")


# ---- Conditional GAN: کنترل خروجی ----
print("\n=== معرفی Conditional GAN (cGAN) ===")
print("""
در GAN معمولی، هیچ کنترلی بر نوع تصویر تولیدشده نداریم.
Conditional GAN با دریافت یک برچسب y (مثل رقم ۰ تا ۹) کنار بردار نویز،
به Generator اجازه می‌دهد تصویر مربوط به آن کلاس را تولید کند.

فرمول cGAN:
min_G max_D V(D,G) = E[log D(x|y)] + E[log(1 - D(G(z|y)|y))]

پیاده‌سازی ساده: برچسب y را به‌صورت One-Hot به ورودی G و D اضافه می‌کنیم.
""")

# نمایش ساختار cGAN Generator
class ConditionalGenerator(nn.Module):
    """Generator شرطی که رقم مورد نظر را دریافت می‌کند"""

    def __init__(self, latent_dim=100, num_classes=10):
        super().__init__()
        # ترکیب بردار نویز با Embedding کلاس
        self.class_embed = nn.Embedding(num_classes, 10)
        self.network = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + 10, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, class_labels):
        class_emb = self.class_embed(class_labels)
        # ترکیب نویز و اطلاعات کلاس
        combined = torch.cat([z, class_emb], dim=1)
        return self.network(combined.unsqueeze(-1).unsqueeze(-1))

cG = ConditionalGenerator().to(device)
z_sample = torch.randn(5, 100, device=device)
labels = torch.tensor([0, 1, 2, 3, 4], device=device)
output = cG(z_sample, labels)
print(f"شکل خروجی cGAN Generator: {output.shape}")
print("با این معماری می‌توان دقیقاً مشخص کرد کدام رقم تولید شود.")
```

---

# ❓ سؤالات تستی

### سؤال ۱

در آموزش GAN، چرا باید Generator و Discriminator به‌صورت متناوب آموزش ببینند و نه همزمان؟

الف) چون PyTorch از آموزش همزمان پشتیبانی نمی‌کند

ب) چون آموزش همزمان باعث می‌شود گرادیان‌های هر دو شبکه با هم تداخل پیدا کنند و آموزش ناپایدار شود؛ در آموزش متناوب، هنگام به‌روزرسانی G، وزن‌های D ثابت است و بالعکس

ج) چون Discriminator همیشه باید قبل از Generator آموزش ببیند

د) چون دو بهینه‌ساز جداگانه نمی‌توانند همزمان اجرا شوند

### سؤال ۲

چرا در DCGAN از LeakyReLU در Discriminator به‌جای ReLU استفاده می‌شود؟

الف) چون LeakyReLU سریع‌تر محاسبه می‌شود

ب) چون ReLU برای مقادیر منفی گرادیان صفر می‌دهد (Dying ReLU) که در Discriminator مشکل‌ساز است؛ LeakyReLU با دادن گرادیان کوچک به مقادیر منفی این مشکل را حل می‌کند

ج) چون تصاویر ورودی Discriminator همیشه منفی هستند

د) چون Batch Normalization فقط با LeakyReLU سازگار است

---

## ✅ پاسخ‌ها

**پاسخ سؤال ۱: گزینه ب**

در هر گام آموزش، باید یکی از دو شبکه ثابت باشد. وقتی G را آموزش می‌دهیم، D باید ثابت باشد تا گرادیان‌های G معنادار باشند. اگر هر دو همزمان تغییر کنند، هدفی که G سعی دارد به آن برسد هم مدام در حال تغییر است که آموزش را به‌شدت ناپایدار می‌کند.

**پاسخ سؤال ۲: گزینه ب**

ReLU برای هر ورودی منفی صفر برمی‌گرداند و مشتق آن هم صفر است. در Discriminator که با تصاویر نرمال‌شده با مقادیر منفی کار می‌کند، استفاده از ReLU باعث می‌شود نورون‌های زیادی «بمیرند» (Dying ReLU). LeakyReLU با ضریب کوچکی مثل ۰.۲ برای مقادیر منفی گرادیان غیرصفر حفظ می‌کند.

---

# 📝 خلاصه فصل

در این فصل با معماری GAN و ایده‌ی بازی minimax میان Generator و Discriminator آشنا شدیم. فرمول ریاضی کامل GAN را بررسی کردیم و یک گام آموزشی کامل را با اعداد واقعی دنبال کردیم. چالش‌های اصلی آموزش GAN مثل Mode Collapse و ناپایداری را شناختیم. معماری DCGAN را که اولین GAN پایدار برای تصویر بود با تمام نوآوری‌هایش بررسی کردیم. در پروژه‌ی عملی یک DCGAN کامل روی MNIST پیاده‌سازی کردیم و در انتها با ایده‌ی Conditional GAN که کنترل خروجی Generator را ممکن می‌کند آشنا شدیم.

---

# 🎯 تمرین‌های پایان فصل

۱. DCGAN را روی CIFAR-10 آموزش دهید و کیفیت تصاویر رنگی تولیدشده را بررسی کنید.
۲. Conditional GAN کامل را پیاده‌سازی کنید که بتواند هر رقم مشخصی را تولید کند.
۳. Mode Collapse را شبیه‌سازی کنید: نرخ یادگیری Discriminator را بسیار بزرگ کنید و ببینید چه اتفاقی می‌افتد.
۴. Label Smoothing را از ۰.۹ به ۱.۰ تغییر دهید و اثر آن بر پایداری آموزش را مقایسه کنید.
۵. معماری Wasserstein GAN (WGAN) را مطالعه کنید که یک معیار بهتر برای اندازه‌گیری فاصله بین توزیع‌ها دارد.

---

در فصل ۲۹ با **یادگیری انتقالی پیشرفته در بینایی ماشین** آشنا می‌شویم و استراتژی‌های Freeze و Fine-tuning تدریجی را در پروژه‌های واقعی تصویری بررسی می‌کنیم.
