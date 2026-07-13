# فصل ۲۷  Autoencoder و Variational Autoencoder (VAE)

## 🎯 اهداف فصل

تا اینجا تمام مدل‌هایی که بررسی کردیم از نوع تبعی (Discriminative) بودند؛ یعنی ورودی را به یک برچسب یا مقدار عددی نگاشت می‌دادند. در بخش پنجم کتاب وارد دنیای مدل‌های مولد (Generative Models) می‌شویم که می‌توانند داده‌ی جدید بسازند. اولین معماری این بخش Autoencoder است که می‌تواند نمایش فشرده‌ای از داده بیاموزد. در پایان این فصل خواهید توانست:

- ساختار **Encoder-Decoder** و مفهوم **فضای نهفته** را توضیح دهید،
- تفاوت **Autoencoder معمولی** از **VAE** را بشناسید،
- فرمول **ELBO در VAE** را با مثال درک کنید،
- هر دو معماری را روی داده‌ی واقعی پیاده‌سازی کنید.

---

# 🔑 ایده‌ی اصلی Autoencoder

Autoencoder یک شبکه‌ی عصبی است که می‌آموزد ورودی را در یک بردار فشرده (کد نهفته) خلاصه کند و سپس آن را بازسازی کند. معماری آن از دو بخش تشکیل شده است:

- **Encoder** که ورودی را فشرده می‌کند
- **Decoder** که از کد فشرده، ورودی را بازسازی می‌کند

تابع هزینه، خطای بازسازی است که اندازه می‌گیرد خروجی مدل چقدر شبیه ورودی اصلی است.

📷 [تصویر اینجا قرار گیرد: نمودار Autoencoder با Encoder که ورودی را به فضای نهفته فشرده می‌کند و Decoder که آن را بازسازی می‌کند، با نمایش تنگه‌ی اطلاعاتی در وسط]

---

# 🎲 Variational Autoencoder: از نقطه به توزیع

مشکل اصلی Autoencoder معمولی این است که فضای نهفته آن ناپیوسته و نامنظم است. یعنی نمی‌توان یک نقطه‌ی تصادفی از این فضا انتخاب کرد و انتظار داشت Decoder چیز معنیداری بسازد.

VAE این مشکل را حل می‌کند: به‌جای یادگیری یک نقطه‌ی ثابت در فضای نهفته، یاد می‌گیرد یک توزیع احتمال (معمولاً گاوسی) را نمایش دهد. Encoder مقادیر **μ** (میانگین) و **log(σ²)** (لگاریتم واریانس) را تولید می‌کند و نمونه‌برداری از این توزیع انجام می‌شود.

فرمول نمونه‌برداری با ترفند Reparameterization:

```
z = μ + σ × ε    که در آن ε ~ N(0, 1)
```

این ترفند اجازه می‌دهد گرادیان از z به μ و σ منتقل شود، چون ε یک متغیر تصادفی مستقل است.

تابع هزینه‌ی VAE ترکیبی از دو جزء است:

```
L = خطای_بازسازی + KL_Divergence
  = E[log p(x|z)] - KL(q(z|x) || p(z))
```

جزء اول (خطای بازسازی) می‌خواهد Decoder تصویر را خوب بازسازی کند. جزء دوم (KL Divergence) می‌خواهد توزیع نهفته به توزیع نرمال استاندارد نزدیک باشد که فضای نهفته را منظم‌تر می‌کند.

---

# 💻 پروژه‌ی عملی: Autoencoder و VAE روی MNIST

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

# بارگذاری MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

trainset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

# ---- Autoencoder معمولی ----
class Autoencoder(nn.Module):

    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# ---- VAE ----
class VAE(nn.Module):

    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()
        # Encoder: تولید μ و log(σ²)
        self.encoder_shared = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder_shared(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def vae_loss(self, recon_x, x, mu, logvar):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + kl_loss) / x.size(0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# آموزش Autoencoder
ae_model = Autoencoder().to(device)
ae_optimizer = optim.Adam(ae_model.parameters(), lr=1e-3)

print("آموزش Autoencoder...")
for epoch in range(10):
    total_loss = 0
    for x, _ in trainloader:
        x = x.to(device)
        ae_optimizer.zero_grad()
        recon, _ = ae_model(x)
        loss = F.mse_loss(recon, x)
        loss.backward()
        ae_optimizer.step()
        total_loss += loss.item()
    if epoch % 3 == 0:
        print(f"  دوره {epoch+1}: خطای بازسازی = {total_loss/len(trainloader):.4f}")

# آموزش VAE
vae_model = VAE().to(device)
vae_optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)

print("\nآموزش VAE...")
for epoch in range(10):
    total_loss = 0
    for x, _ in trainloader:
        x = x.to(device)
        vae_optimizer.zero_grad()
        recon, mu, logvar = vae_model(x)
        loss = vae_model.vae_loss(recon, x, mu, logvar)
        loss.backward()
        vae_optimizer.step()
        total_loss += loss.item()
    if epoch % 3 == 0:
        print(f"  دوره {epoch+1}: خطای کل (Recon+KL) = {total_loss/len(trainloader):.4f}")

# تولید تصاویر جدید با VAE
print("\nتولید تصاویر جدید با نمونه‌برداری از فضای نهفته:")
vae_model.eval()
with torch.no_grad():
    z_random = torch.randn(10, 32).to(device)
    generated = vae_model.decoder(z_random)
    print(f"شکل تصاویر تولیدشده: {generated.shape}")
    print("تصاویر با موفقیت تولید شدند (هر ردیف یک رقم تولیدی)")

# آزمایش درون‌یابی در فضای نهفته
print("\nدرون‌یابی بین دو نمونه در فضای نهفته:")
test_imgs, _ = next(iter(testloader))
with torch.no_grad():
    mu1, _ = vae_model.encode(test_imgs[0:1].to(device))
    mu2, _ = vae_model.encode(test_imgs[1:2].to(device))
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    for alpha in alphas:
        z_interp = (1 - alpha) * mu1 + alpha * mu2
        img = vae_model.decoder(z_interp)
        print(f"  α={alpha}: تصویر میان‌یابی‌شده ساخته شد")
```

---

# ❓ سؤالات تستی

## سؤال ۱

چرا در VAE از ترفند Reparameterization استفاده می‌شود؟

الف) برای کاهش تعداد پارامترها  
ب) چون نمونه‌برداری مستقیم از توزیع تصادفی، گرادیان را قطع می‌کند و ترفند Reparameterization با جدا کردن تصادفی‌بودن به ε، امکان انتقال گرادیان به μ و σ را فراهم می‌کند  
ج) برای افزایش سرعت محاسبات  
د) برای کاهش KL Divergence  

## سؤال ۲

KL Divergence در تابع هزینه‌ی VAE چه نقشی دارد؟

الف) خطای بازسازی را اندازه می‌گیرد  
ب) فضای نهفته را منظم می‌کند تا به توزیع نرمال استاندارد نزدیک باشد و نمونه‌برداری تصادفی معنادار باشد  
ج) تعداد پارامترهای مدل را محدود می‌کند  
د) سرعت آموزش را کاهش می‌دهد  

---

# ✅ پاسخ‌ها

## پاسخ سؤال ۱: گزینه ب

بدون ترفند Reparameterization، نمونه‌برداری z از توزیع N(μ, σ²) یک عملیات تصادفی غیرقابل‌مشتق است. با z = μ + σ × ε، تصادفی‌بودن به ε منتقل می‌شود که پارامتر مدل نیست، و گرادیان می‌تواند از z به μ و σ جاری شود.

## پاسخ سؤال ۲: گزینه ب

KL Divergence فاصله بین توزیع نهفته‌ی یادگرفته‌شده q(z|x) و توزیع نرمال استاندارد p(z) را اندازه می‌گیرد و مدل را تشویق می‌کند این فاصله را کوچک کند. این باعث می‌شود فضای نهفته یکپارچه و پیوسته شود.

---

# 📝 خلاصه فصل

Autoencoder با فشردن اطلاعات به یک بردار نهفته، می‌تواند برای کاهش ابعاد، حذف نویز و تشخیص ناهنجاری استفاده شود. VAE با تبدیل فضای نهفته به یک توزیع احتمال منظم، قابلیت تولید نمونه‌های جدید، درون‌یابی در فضای نهفته، و یادگیری نمایش‌های معنادار را اضافه می‌کند. در پروژه‌ی عملی هر دو معماری را روی MNIST آموزش دادیم و قابلیت تولید و درون‌یابی VAE را نشان دادیم.

---

# 🎯 تمرین‌های پایان فصل

۱. یک Autoencoder کانولوشنی (Convolutional Autoencoder) برای CIFAR-10 بسازید.  
۲. فضای نهفته‌ی VAE را با t-SNE در دو بُعد تجسم کنید و ببینید ارقام مختلف چگونه خوشه‌بندی می‌شوند.  
۳. یک Denoising Autoencoder بسازید که ورودی پرنویز دارد اما یاد می‌گیرد تصویر تمیز را بازسازی کند.  
۴. beta-VAE را که ضریب KL Divergence را تغییر می‌دهد امتحان کنید.  
۵. درون‌یابی را بین تمام ۱۰ جفت از ارقام ۰ تا ۹ انجام دهید.  

---

در فصل ۲۸ با **شبکه‌های مولد تخاصمی (GAN)** آشنا می‌شویم.