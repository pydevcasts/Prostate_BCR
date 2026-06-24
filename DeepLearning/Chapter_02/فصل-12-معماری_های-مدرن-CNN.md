# فصل ۱۲ 🧬 معماری‌های مدرن CNN: ResNet، Inception و EfficientNet

## 🎯 اهداف فصل

در فصل قبل دیدیم که VGG با عمیق‌تر کردن شبکه نتایج بهتری گرفت. اما اگر این منطق درست است، آیا می‌توانیم شبکه‌هایی با صد یا هزار لایه بسازیم و نتایج بازهم بهتری بگیریم؟ در عمل جواب منفی است و دلیل آن مشکل معروف گرادیان محوشونده در شبکه‌های خیلی عمیق است. در این فصل می‌بینیم که معماری‌های مدرن چگونه این سد را شکستند. در پایان این فصل خواهید توانست مشکل گرادیان محوشونده در شبکه‌های عمیق را توضیح دهید، ایده‌ی Skip Connection در ResNet را با مثال ریاضی درک کنید، معماری Inception Module را بشناسید، مفهوم Compound Scaling در EfficientNet را بفهمید، و یک ResNet سفارشی را روی دیتاست واقعی پیاده‌سازی کنید.

---

# ⚠️ مشکل گرادیان محوشونده در شبکه‌های خیلی عمیق

زمانی که تعداد لایه‌های یک شبکه از بیست یا سی لایه فراتر می‌رود، پدیده‌ی عجیبی رخ می‌دهد: دقت شبکه نه‌تنها بهبود نمی‌یابد، بلکه بدتر از شبکه‌ی کم‌عمق‌تر می‌شود. این مشکل را تباهی (Degradation Problem) می‌نامند و با بیش‌برازش متفاوت است، چراکه حتی خطای آموزشی هم بدتر می‌شود. دلیل ریشه‌ای این مشکل را می‌توان در پس‌انتشار خطا یافت؛ وقتی گرادیان از لایه‌ی آخر به سمت لایه‌های اولیه باید از ده‌ها لایه عبور کند، در هر عبور با مشتقاتی ضرب می‌شود که اغلب کوچک‌تر از یک هستند. ضرب کردن اعداد کوچک‌تر از یک در هم، به‌سرعت به مقادیر نزدیک به صفر می‌رسد و وقتی گرادیان به لایه‌های ابتدایی می‌رسد، آن‌قدر کوچک شده که عملاً هیچ اصلاحی در وزن‌ها انجام نمی‌شود. به عبارت دیگر، لایه‌های ابتدایی شبکه یاد نمی‌گیرند.

---

# 🔗 ResNet: حل ظریف یک مشکل بزرگ با Skip Connection

در سال دو هزار و پانزده، کای مینگ هی و همکارانش در مایکروسافت، ایده‌ای معرفی کردند که ساده اما تغییردهنده‌ی بازی بود: به‌جای این‌که تنها از یک مسیر عمیق برای انتقال اطلاعات استفاده کنیم، یک مسیر میان‌بر (Skip Connection یا Residual Connection) نیز ایجاد کنیم که ورودی یک بلوک را مستقیماً به خروجی همان بلوک اضافه کند. این ایده به‌ظاهر ساده، مشکل تباهی در شبکه‌های عمیق را به‌طور کامل حل کرد و ResNet توانست شبکه‌هایی با صد و پنجاه و دو لایه (ResNet-152) را با موفقیت آموزش دهد.

برای فهمیدن چرایی این موفقیت، به ریاضیات پشت آن نگاه می‌کنیم. در یک بلوک معمولی، اگر ورودی x باشد و لایه‌های کانولوشنی تابع F(x) را یاد بگیرند، خروجی برابر است با F(x). اما در یک بلوک باقی‌مانده (Residual Block)، خروجی برابر است با F(x) + x، یعنی تابع یادگرفته‌شده به ورودی اصلی اضافه می‌شود:

```
خروجی بلوک معمولی:   H(x) = F(x)
خروجی Residual Block: H(x) = F(x) + x
```

این تفاوت ظاهراً کوچک، تأثیر عمیقی روی پس‌انتشار خطا دارد. وقتی گرادیان از خروجی به سمت ورودی بلوک بازمی‌گردد، به دلیل وجود Skip Connection، گرادیان از دو مسیر موازی منتقل می‌شود: مسیر اصلی از طریق F(x)، و مسیر میان‌بر که مستقیماً گرادیان را بدون هیچ تغییری منتقل می‌کند. این مسیر مستقیم تضمین می‌کند که حتی اگر مسیر اصلی دچار محوشدگی گرادیان شود، گرادیان هنوز از طریق Skip Connection می‌تواند به لایه‌های ابتدایی برسد.

یک مثال عددی ساده این تأثیر را روشن می‌کند. فرض کنید گرادیان در مسیر اصلی در هر لایه با ضریب نه دهم ضرب می‌شود. پس از ده لایه، گرادیان برابر نه دهم به توان ده می‌شود که تقریباً برابر با سی و پنج صدم است. اما با Skip Connection، گرادیان کل برابر است با حداقل ۱ (از مسیر میان‌بر) به‌علاوه‌ی مقدار مسیر اصلی. حتی اگر مسیر اصلی صفر شود، گرادیان هنوز از مسیر میان‌بر برابر با یک منتقل می‌شود.

📷 [تصویر اینجا قرار گیرد: نمودار یک Residual Block با نمایش مسیر اصلی F(x) و مسیر میان‌بر Skip Connection که ورودی x را مستقیماً به خروجی اضافه می‌کند]

---

# 🏗️ ساختار کامل ResNet

ResNet از پشته‌هایی از Residual Block تشکیل شده است. هر Residual Block معمولاً از دو یا سه لایه‌ی کانولوشنی با Batch Normalization و ReLU تشکیل می‌شود. برای نسخه‌های عمیق‌تر ResNet (ResNet-50 به بالا)، از یک نوع بلوک بهینه‌تر به نام Bottleneck Block استفاده می‌شود که با استفاده از کانولوشن ۱×۱ ابتدا تعداد کانال‌ها را کاهش می‌دهد، سپس کانولوشن ۳×۳ اصلی را اعمال می‌کند، و در انتها دوباره با کانولوشن ۱×۱ تعداد کانال‌ها را افزایش می‌دهد. این طراحی تعداد پارامترها را بدون کاهش عملکرد کاهش می‌دهد. نسخه‌های مختلف ResNet بر اساس تعداد کل لایه نام‌گذاری شده‌اند: ResNet-18، ResNet-34، ResNet-50، ResNet-101 و ResNet-152.

📷 [تصویر اینجا قرار گیرد: مقایسه‌ی یک Basic Residual Block (دو لایه ۳×۳) با یک Bottleneck Block (کانولوشن ۱×۱، ۳×۳، ۱×۱) با نمایش Skip Connection در هر دو]

---

# 🌟 Inception/GoogLeNet: چند مقیاس به‌صورت موازی

در همان سال ۲۰۱۴، گوگل معماری دیگری به نام GoogLeNet یا Inception معرفی کرد که از زاویه‌ی متفاوتی به مشکل عمق شبکه نگاه می‌کرد. سؤال اصلی طراحان Inception این بود: به‌جای این‌که فقط یک اندازه فیلتر در هر لایه استفاده کنیم، چه می‌شود اگر همزمان چندین اندازه‌ی فیلتر مختلف را اعمال کنیم و نتایج را کنار هم قرار دهیم؟ این ایده به Inception Module تبدیل شد که در آن، ورودی به‌طور موازی از چهار مسیر عبور می‌کند: یک کانولوشن ۱×۱، یک کانولوشن ۳×۳، یک کانولوشن ۵×۵، و یک Max Pooling. سپس خروجی همه‌ی این مسیرها در امتداد محور کانال به هم متصل می‌شوند.

منطق این طراحی این است که ویژگی‌های مختلف تصویر در مقیاس‌های متفاوت اتفاق می‌افتند؛ لبه‌های ظریف با فیلترهای کوچک بهتر تشخیص داده می‌شوند، در حالی که الگوهای بزرگ‌تر به فیلترهای بزرگ‌تر نیاز دارند. با استفاده‌ی همزمان از هر دو، شبکه می‌تواند بهترین ویژگی را از هر مقیاس انتخاب کند. برای کاهش هزینه‌ی محاسباتی، از کانولوشن‌های ۱×۱ پیش از هر کانولوشن بزرگ‌تر به‌عنوان کاهنده‌ی بُعد (Dimensionality Reduction) استفاده می‌شود.

📷 [تصویر اینجا قرار گیرد: نمودار یک Inception Module با نمایش چهار مسیر موازی (1×1، 3×3، 5×5 و Pooling) و اتصال خروجی‌ها در محور کانال]

---

# ⚡ EfficientNet: مقیاس‌بندی هوشمند

در سال دو هزار و نوزده، گوگل مقاله‌ای منتشر کرد که یک سؤال اساسی می‌پرسید: روش بهینه برای افزایش اندازه‌ی یک CNN چیست؟ سه روش معمول برای بزرگ‌تر کردن یک شبکه وجود دارد: افزایش عمق (تعداد لایه‌ها)، افزایش عرض (تعداد کانال‌ها) یا افزایش وضوح (اندازه‌ی تصویر ورودی). EfficientNet نشان داد که بهترین نتیجه از ترکیب هر سه این روش‌ها با نسبت‌های مشخص به دست می‌آید و این رویکرد را Compound Scaling نامید.

نتیجه‌ی این رویکرد شگفت‌انگیز بود: EfficientNet-B7 روی ImageNet به دقت ۸۴.۳ درصد رسید، در حالی که تنها ۶۶ میلیون پارامتر داشت؛ در مقایسه ResNet-152 با ۶۰ میلیون پارامتر به دقت ۷۷.۸ درصد می‌رسید. این یعنی EfficientNet با پارامتر مشابه، بیش از ۶ درصد دقت بهتری داشت. این معماری نشان داد که معماری پایه و روش مقیاس‌بندی، هر دو به اندازه‌ی عمق شبکه اهمیت دارند.

---

# 💻 پروژه‌ی عملی فصل: پیاده‌سازی ResNet سفارشی روی CIFAR-10

در این پروژه یک ResNet با Residual Block واقعی از صفر پیاده‌سازی می‌کنیم. هدف این است که درک عمیقی از پیاده‌سازی Skip Connection در PyTorch به دست آورید؛ یعنی ببینید که وقتی ابعاد ورودی و خروجی یک بلوک متفاوت هستند، چگونه باید Skip Connection را تطبیق داد.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


class ResidualBlock(nn.Module):
    """
    یک Residual Block با دو لایه‌ی کانولوشنی ۳×۳.
    اگر تعداد کانال‌های ورودی و خروجی متفاوت باشند،
    از یک کانولوشن ۱×۱ برای تطبیق Skip Connection استفاده می‌شود.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # مسیر اصلی: دو کانولوشن ۳×۳
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # مسیر میان‌بر (Skip Connection)
        # اگر ابعاد ورودی و خروجی متفاوتند، نیاز به تطبیق داریم
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # ابعاد یکسان است؛ ورودی بدون تغییر اضافه می‌شود
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # خروجی مسیر اصلی به‌علاوه‌ی ورودی اصلی از مسیر میان‌بر
        out = self.main_path(x) + self.shortcut(x)
        return self.relu(out)


class ResNet_Custom(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        # لایه اولیه: کانولوشن ابتدایی
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # چهار گروه از Residual Blockها
        # گروه اول: ۶۴ کانال، بدون کاهش ابعاد
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )

        # گروه دوم: ۱۲۸ کانال، کاهش ابعاد با stride=2
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )

        # گروه سوم: ۲۵۶ کانال، کاهش ابعاد با stride=2
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )

        # گروه چهارم: ۵۱۲ کانال، کاهش ابعاد با stride=2
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512)
        )

        # Global Average Pooling و طبقه‌بند
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet_Custom().to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"تعداد پارامترهای ResNet سفارشی: {total_params:,}")

# تأیید جریان داده با یک ورودی آزمایشی
dummy = torch.randn(1, 3, 32, 32).to(device)
output = model(dummy)
print(f"شکل خروجی برای یک تصویر آزمایشی: {output.shape}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)
# زمان‌بند Cosine که در پژوهش‌های ResNet بسیار رایج است
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

best_acc = 0.0

for epoch in range(100):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images), 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = 100 * test_correct / test_total
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_resnet_cifar10.pt')

    scheduler.step()

    if epoch % 10 == 0 or epoch == 99:
        print(f"دوره {epoch+1:3d} | "
              f"خطا {running_loss/len(trainloader):.3f} | "
              f"آموزش {train_acc:.1f}% | "
              f"آزمون {test_acc:.1f}% | "
              f"بهترین {best_acc:.1f}%")

print(f"\nآموزش کامل شد. بهترین دقت آزمون: {best_acc:.2f}%")
```

چند نکته‌ی مهم در این پیاده‌سازی وجود دارد که باید به آن‌ها توجه کنید. نخست، در کلاس ResidualBlock، وقتی stride برابر ۲ است یا تعداد کانال‌های ورودی و خروجی متفاوت است، از یک کانولوشن ۱×۱ برای تطبیق ابعاد Skip Connection استفاده می‌شود. اگر این تطبیق انجام نشود، جمع کردن ورودی با خروجی به خطای ابعاد می‌انجامد. دوم، از `nn.AdaptiveAvgPool2d(1)` به‌جای Max Pooling ثابت استفاده شده که ابعاد نقشه‌های ویژگی را به ۱×۱ کاهش می‌دهد؛ این رویکرد Global Average Pooling که از معماری‌های Inception و ResNet الهام گرفته، تعداد پارامترهای طبقه‌بند را به‌شدت کاهش می‌دهد. سوم، برخلاف فصل‌های قبل که از Adam استفاده کردیم، اینجا از SGD با Momentum استفاده شده؛ در پژوهش‌های اصلی ResNet این ترکیب نتایج بهتری داده و هنوز هم در بسیاری از مقالات پژوهشی استفاده می‌شود.

---

# ❓ سؤالات تستی

### سؤال ۱

چرا در یک Residual Block، وقتی stride برابر ۲ است، باید Skip Connection هم با یک کانولوشن ۱×۱ تطبیق پیدا کند؟

الف) چون کانولوشن ۱×۱ سریع‌تر است

ب) چون ابعاد فضایی خروجی مسیر اصلی (با stride=2) از ابعاد ورودی کوچک‌تر است و برای جمع کردن، هر دو باید ابعاد یکسانی داشته باشند

ج) چون PyTorch از جمع کردن تنسورهای بزرگ پشتیبانی نمی‌کند

د) چون Skip Connection فقط در لایه‌های آخر کاربرد دارد

### سؤال ۲

ایده‌ی Compound Scaling در EfficientNet چیست؟

الف) افزایش فقط عمق شبکه برای بهبود دقت

ب) کاهش همزمان عمق، عرض و وضوح برای کاهش پارامترها

ج) افزایش همزمان عمق، عرض و وضوح با نسبت‌های بهینه به‌جای افزایش تنها یک بُعد

د) استفاده از Skip Connection در تمام لایه‌ها

---

## ✅ پاسخ‌ها

**پاسخ سؤال ۱: گزینه ب**

وقتی stride=2 در مسیر اصلی استفاده می‌شود، ابعاد فضایی خروجی نصف ورودی می‌شود. برای این‌که بتوانیم این خروجی را با ورودی اصلی (که ابعاد بزرگ‌تری دارد) جمع کنیم، باید با کانولوشن ۱×۱ با stride=2 ابعاد Skip Connection را هم نصف کنیم تا ابعاد یکسان شوند.

**پاسخ سؤال ۲: گزینه ج**

Compound Scaling نشان داد که بهترین نتیجه از افزایش هر سه بُعد عمق، عرض و وضوح با نسبت‌های مشخص و متعادل به دست می‌آید. این رویکرد به EfficientNet امکان داد با پارامتر کمتر از رقبا دقت بالاتری به دست آورد.

---

# 📝 خلاصه فصل

در این فصل با سه معماری مدرن و تأثیرگذار CNN آشنا شدیم. ResNet با ایده‌ی Skip Connection مشکل تباهی در شبکه‌های خیلی عمیق را حل کرد و آموزش شبکه‌هایی با صدها لایه را ممکن ساخت؛ منطق ریاضی این ایده آن است که گرادیان از مسیر میان‌بر حتی اگر مسیر اصلی دچار محوشدگی شود همچنان جریان دارد. Inception Module با پردازش موازی در چند مقیاس مختلف، به شبکه امکان می‌دهد ویژگی‌هایی با اندازه‌های مختلف را همزمان یاد بگیرد. EfficientNet با ایده‌ی Compound Scaling نشان داد که مقیاس‌بندی متعادل هر سه بُعد شبکه بهتر از افزایش تنها یک بُعد است. در پروژه‌ی عملی یک ResNet سفارشی کامل با Residual Block واقعی پیاده‌سازی کردیم.

---

# 🎯 تمرین‌های پایان فصل

۱. یک Residual Block با سه لایه‌ی کانولوشنی (Bottleneck Block) به‌جای دو لایه پیاده‌سازی کنید.

۲. نسخه‌ی از‌پیش‌آموزش‌دیده‌ی ResNet-18 را از torchvision بارگذاری کنید و لایه‌ی آخر آن را برای CIFAR-10 تنظیم کنید.

۳. یک Inception Module ساده با سه مسیر موازی (1×1، 3×3 و Max Pooling) در PyTorch پیاده‌سازی کنید.

۴. تأثیر حذف Skip Connection از ResNet_Custom را با نگاه به روند تغییرات خطای آموزش بررسی کنید.

۵. با استفاده از کتابخانه‌ی timm که مجموعه‌ای از معماری‌های از‌پیش‌آموزش‌دیده را فراهم می‌کند، یک EfficientNet-B0 را برای CIFAR-10 آزمایش کنید.

---

در فصل ۱۳ به **پروژه‌ی جامع بینایی ماشین** می‌رسیم؛ جایی که تمام مفاهیم بخش دوم را در یک پروژه‌ی کامل تشخیص بیماری پوستی از تصاویر پزشکی ترکیب می‌کنیم و از Transfer Learning برای رسیدن به دقتی حرفه‌ای استفاده می‌کنیم.
