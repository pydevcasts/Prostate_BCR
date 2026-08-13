# فصل ۱۱ 🏛️ معماری‌های کلاسیک CNN: LeNet، AlexNet و VGG

## 🎯 اهداف فصل

در فصل‌های قبل با ساختار پایه‌ی CNN آشنا شدیم و یک مدل سه‌بلوکی ساده طراحی کردیم. اما در دنیای واقعی، معماری‌هایی وجود دارند که هرکدام در زمان خود نقطه‌ی عطف مهمی در تاریخ یادگیری عمیق بوده‌اند و درک آن‌ها دو فایده دارد: اول، تاریخچه‌ی تکاملی ایده‌ها را می‌فهمیم و می‌بینیم که هر نسل معماری چه مشکلی را از نسل قبل حل کرده است؛ دوم، بسیاری از ایده‌های این معماری‌های کلاسیک هنوز در شبکه‌های مدرن استفاده می‌شوند. در پایان این فصل خواهید توانست معماری LeNet، AlexNet و VGG را با تمام جزئیاتشان توضیح دهید، نوآوری اصلی هرکدام را بشناسید، و هر سه را در PyTorch پیاده‌سازی و مقایسه کنید.

---

# 🌱 LeNet-5: اولین CNN عملی تاریخ (1998)

داستان شبکه‌های کانولوشنی مدرن با یان لکون (Yann LeCun) در سال هزار و نهصد و نود و هشت آغاز می‌شود. لکون و همکارانش در آزمایشگاه‌های Bell، شبکه‌ای به نام LeNet-5 طراحی کردند که برای تشخیص ارقام دست‌نویس در چک‌های بانکی به کار می‌رفت. این شبکه برای اولین بار نشان داد که یک CNN می‌تواند در یک کاربرد تجاری واقعی عملکرد بسیار خوبی داشته باشد و در واقع اولین استقرار موفق یادگیری عمیق در صنعت بود.

معماری LeNet-5 بسیار ساده است و از دو بلوک کانولوشنی و سه لایه‌ی تمام‌متصل تشکیل شده است. تصویر ورودی آن ۳۲ در ۳۲ پیکسل سیاه‌وسفید است. بلوک اول شامل یک لایه‌ی کانولوشنی با ۶ فیلتر ۵ در ۵ و یک لایه‌ی Average Pooling 2×2 است. بلوک دوم شامل یک لایه‌ی کانولوشنی با ۱۶ فیلتر ۵ در ۵ و یک Average Pooling دیگر است. پس از Flatten، سه لایه‌ی تمام‌متصل با صد و بیست، هشتاد و چهار و ده نورون (برای ده رقم دست‌نویس) خروجی نهایی را تولید می‌کنند. دو نوآوری مهم LeNet عبارتند از: استفاده از کانولوشن به‌جای اتصال کامل برای کاهش پارامترها، و استفاده از Pooling برای ایجاد ثبات موقعیتی. هر دوی این ایده‌ها امروزه در تمام معماری‌های CNN دیده می‌شوند.

📷 [تصویر اینجا قرار گیرد: معماری کامل LeNet-5 با نمایش ابعاد داده در هر مرحله از ورودی ۳۲×۳۲ تا خروجی ۱۰ کلاسه]

---

# 💥 AlexNet: انقلاب یادگیری عمیق (2012)

اگر LeNet نقطه‌ی شروع بود، AlexNet نقطه‌ی انقلاب بود. در سال دو هزار و دوازده، آلکس کریژوسکی، ایلیا ساتسکیور و جفری هینتون در مسابقه‌ی بینایی ماشین ImageNet که بزرگترین رقابت سالانه در این حوزه بود، با نرخ خطایی که ۱۱ درصد از بهترین روش‌های قبلی کمتر بود، اول شدند. این فاصله‌ی عظیم نشان می‌داد که یادگیری عمیق نه تنها کمی بهتر، بلکه یک دسته بهتر از سایر روش‌ها عمل می‌کند و باعث شد کل جامعه‌ی علمی بینایی ماشین به‌سرعت به سمت یادگیری عمیق متمایل شود.

AlexNet از پنج لایه‌ی کانولوشنی و سه لایه‌ی تمام‌متصل تشکیل شده است و روی تصاویر رنگی ۲۲۴ در ۲۲۴ پیکسل از هزار کلاس مختلف (دیتاست ImageNet) کار می‌کند. تعداد پارامترهای آن حدود شصت میلیون است که در مقایسه با LeNet جهشی عظیم محسوب می‌شود. نوآوری‌های اصلی AlexNet که همگی هنوز هم در معماری‌های مدرن استفاده می‌شوند عبارتند از: نخست، استفاده‌ی گسترده از ReLU به‌جای Sigmoid که آموزش را بسیار سریع‌تر کرد؛ دوم، استفاده از Dropout برای اولین بار در یک معماری بزرگ برای جلوگیری از بیش‌برازش؛ سوم، استفاده از Data Augmentation (برش تصادفی و آینه‌ای کردن) برای افزایش حجم مؤثر داده؛ و چهارم، آموزش موازی روی دو GPU که در آن زمان نوآوری محسوب می‌شد.

📷 [تصویر اینجا قرار گیرد: معماری AlexNet با نمایش پنج لایه‌ی کانولوشنی و سه لایه‌ی تمام‌متصل و ابعاد هر لایه]

---

# 🏗️ VGG: عمق، ساده‌ترین کلید موفقیت (2014)

در سال دو هزار و چهارده، گروه Visual Geometry Group از دانشگاه آکسفورد (که نام VGG از همین گروه گرفته شده) با ایده‌ی جذاب و ساده‌ای وارد رقابت ImageNet شد: آیا اگر به‌جای استفاده از فیلترهای بزرگ (مثل ۷×۷ یا ۵×۵ در AlexNet)، فقط از فیلترهای ۳×۳ کوچک اما در لایه‌های خیلی عمیق‌تر استفاده کنیم، نتیجه بهتر می‌شود؟ پاسخ مثبت بود و VGG نشان داد که عمق شبکه، نه اندازه‌ی فیلترها، کلید اصلی موفقیت است.

دلیل ریاضی برتری دو لایه‌ی ۳×۳ نسبت به یک لایه‌ی ۵×۵ را می‌توان در دو مزیت خلاصه کرد. اول، میدان پذیرنده‌ی مؤثر دو لایه‌ی ۳×۳ پشت‌سرهم برابر ۵×۵ است، یعنی از نظر بینایی همان حجم از اطلاعات تصویر را می‌بینند. اما از نظر پارامتر، دو لایه‌ی ۳×۳ (هرکدام ۹ وزن) جمعاً ۱۸ وزن دارند، در حالی که یک لایه‌ی ۵×۵ تنها ۲۵ وزن دارد. دوم، بین دو لایه‌ی ۳×۳ می‌توان یک تابع فعال‌سازی اضافه کرد که غیرخطی‌بودن بیشتری به شبکه می‌دهد. VGG در دو نسخه‌ی اصلی معرفی شد: VGG-16 با شانزده لایه‌ی قابل‌یادگیری و VGG-19 با نوزده لایه. هر دو از همان ساختار تکرارشونده‌ی «چند کانولوشن ۳×۳ پشت‌سرهم + Max Pooling» استفاده می‌کنند.

📷 [تصویر اینجا قرار گیرد: معماری VGG-16 با نمایش پنج بلوک کانولوشنی و سه لایه‌ی تمام‌متصل و ابعاد داده در هر بلوک]

---

# 🔢 مقایسه‌ی کمّی سه معماری

| ویژگی | LeNet-5 | AlexNet | VGG-16 |
|---|---|---|---|
| سال | ۱۹۹۸ | ۲۰۱۲ | ۲۰۱۴ |
| تعداد لایه | ۷ | ۸ | ۱۶ |
| تعداد پارامتر | ۶۰ هزار | ۶۰ میلیون | ۱۳۸ میلیون |
| اندازه‌ی فیلتر | ۵×۵ | ۱۱×۱۱ تا ۳×۳ | فقط ۳×۳ |
| تابع فعال | Tanh | ReLU | ReLU |
| نرخ خطا ImageNet | — | ۱۵٪ | ۷٪ |

---

# 💻 پروژه‌ی عملی فصل: پیاده‌سازی و مقایسه‌ی LeNet، AlexNet کوچک و VGG کوچک

در این پروژه هر سه معماری را (با نسخه‌های کوچک‌تر برای تناسب با CIFAR-10) پیاده‌سازی و عملکرد آن‌ها را روی همان دیتاست مقایسه می‌کنیم. این مقایسه به‌خوبی نشان می‌دهد که چگونه هر نسل معماری عملکرد بهتری نسبت به نسل قبل دارد.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# آماده‌سازی دیتاست CIFAR-10
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


# ---- معماری LeNet برای CIFAR-10 ----
# تصاویر CIFAR-10 رنگی هستند پس کانال ورودی ۳ است نه ۱
class LeNet_CIFAR(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # بلوک اول: (3,32,32) → (6,14,14)
            nn.Conv2d(3, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            # بلوک دوم: (6,14,14) → (16,5,5)
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---- معماری AlexNet کوچک برای CIFAR-10 ----
class AlexNet_Small(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # بلوک اول: استفاده از فیلتر بزرگ‌تر برای اولین لایه
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # بلوک دوم
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # بلوک سوم: سه لایه کانولوشنی متوالی بدون Pooling
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---- معماری VGG کوچک برای CIFAR-10 ----
# استفاده از اصل کلیدی VGG: فقط فیلتر ۳×۳ در لایه‌های عمیق
class VGG_Small(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # بلوک ۱: ۲ لایه کانولوشنی + Pooling
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),            # → (64, 16, 16)

            # بلوک ۲: ۲ لایه کانولوشنی + Pooling
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),            # → (128, 8, 8)

            # بلوک ۳: ۳ لایه کانولوشنی + Pooling (الگوی VGG)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)             # → (256, 4, 4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_and_evaluate(model, model_name, epochs=20):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0.0

    for epoch in range(epochs):

        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(model(images), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc
        scheduler.step()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nمعماری {model_name}:")
    print(f"  تعداد پارامتر: {n_params:,}")
    print(f"  بهترین دقت روی آزمون: {best_acc:.2f}%")
    return best_acc


print("=== مقایسه‌ی سه معماری کلاسیک روی CIFAR-10 ===\n")

acc_lenet = train_and_evaluate(LeNet_CIFAR(), "LeNet")
acc_alexnet = train_and_evaluate(AlexNet_Small(), "AlexNet کوچک")
acc_vgg = train_and_evaluate(VGG_Small(), "VGG کوچک")

print(f"\n{'='*40}")
print(f"خلاصه‌ی مقایسه:")
print(f"  LeNet:        {acc_lenet:.2f}%")
print(f"  AlexNet کوچک: {acc_alexnet:.2f}%")
print(f"  VGG کوچک:     {acc_vgg:.2f}%")
print(f"{'='*40}")
```

نتیجه‌ی این مقایسه‌ی مستقیم، روایت تکاملی معماری‌های CNN را به‌خوبی نشان می‌دهد. LeNet با پارامترهای بسیار کم‌تر، دقت پایین‌تری دارد زیرا ظرفیت یادگیری آن برای تصاویر رنگی پیچیده‌ی CIFAR-10 کافی نیست. AlexNet کوچک با استفاده از ReLU و Dropout دقت بهتری به دست می‌آورد. VGG کوچک با اصل فیلترهای ۳×۳ در معماری عمیق‌تر، بهترین عملکرد را نشان می‌دهد. این مشاهده عملاً همان درسی است که جامعه‌ی یادگیری عمیق بین سال‌های ۱۹۹۸ تا ۲۰۱۴ آموخت و هر پیشرفت، با یک ایده‌ی ساده اما مهم همراه بود.

---

# ❓ سؤالات تستی

### سؤال ۱

کدام نوآوری AlexNet بیشترین تأثیر را بر معماری‌های بعدی CNN داشت؟

الف) استفاده از Average Pooling به‌جای Max Pooling

ب) استفاده از ReLU به‌جای Sigmoid به‌عنوان تابع فعال‌سازی و استفاده از Dropout برای جلوگیری از بیش‌برازش

ج) کاهش تعداد لایه‌ها به پنج لایه

د) استفاده از فیلترهای ۳×۳ در تمام لایه‌ها

### سؤال ۲

چرا VGG برای جایگزینی یک لایه‌ی ۵×۵ از دو لایه‌ی ۳×۳ پشت‌سرهم استفاده کرد؟

الف) چون فیلترهای ۵×۵ در PyTorch پشتیبانی نمی‌شوند

ب) چون دو لایه‌ی ۳×۳ میدان پذیرنده‌ی یکسانی با یک لایه‌ی ۵×۵ دارند اما پارامتر کمتر و غیرخطی‌بودن بیشتری فراهم می‌کنند

ج) چون لایه‌های ۳×۳ سریع‌تر آموزش می‌گیرند

د) چون لایه‌های ۳×۳ از Max Pooling بهتر هستند

---

## ✅ پاسخ‌ها

**پاسخ سؤال ۱: گزینه ب**

ReLU با حل مشکل گرادیان محوشونده در سیگموید، آموزش شبکه‌های عمیق را ممکن کرد و Dropout به‌عنوان روش منظم‌سازی مؤثر معرفی شد. هر دوی این نوآوری‌ها هنوز در اکثر معماری‌های مدرن استفاده می‌شوند.

**پاسخ سؤال ۲: گزینه ب**

میدان پذیرنده‌ی مؤثر دو لایه‌ی ۳×۳ پشت‌سرهم برابر ۵×۵ است. اما از نظر پارامتر، دو لایه‌ی ۳×۳ جمعاً ۱۸ وزن دارند در برابر ۲۵ وزن یک لایه‌ی ۵×۵. علاوه بر این، قرار دادن ReLU بین دو لایه‌ی ۳×۳ باعث می‌شود شبکه غیرخطی‌بودن بیشتری داشته باشد.

---

# 📝 خلاصه فصل

در این فصل سه معماری تاریخی و تأثیرگذار CNN را بررسی کردیم. LeNet نشان داد که CNN می‌تواند در کاربردهای تجاری واقعی موفق باشد. AlexNet با نوآوری‌های ReLU، Dropout و Data Augmentation، انقلاب یادگیری عمیق را کلید زد و ثابت کرد که شبکه‌های عصبی عمیق می‌توانند در مقیاس بزرگ به نتایج باورنکردنی برسند. VGG با اصل ساده‌ی «فقط ۳×۳ اما عمیق‌تر» نشان داد که معماری منظم و اصولی از معماری پیچیده بهتر است. در پروژه‌ی عملی نیز سیر تکاملی این معماری‌ها را با مقایسه‌ی مستقیم روی CIFAR-10 تجربه کردیم.

---

# 🎯 تمرین‌های پایان فصل

۱. تعداد پارامترهای LeNet اصلی (برای تصاویر ۳۲×۳۲ سیاه‌وسفید) را به‌صورت دستی برای هر لایه محاسبه کنید.

۲. نسخه‌ی اصلی VGG از Batch Normalization استفاده نمی‌کرد؛ آن را از مدل VGG_Small حذف کنید و تأثیرش بر سرعت همگرایی را مشاهده کنید.

۳. تعداد لایه‌های VGG_Small را با اضافه کردن یک بلوک چهارم (سه لایه‌ی کانولوشنی ۵۱۲ فیلتره) افزایش دهید.

۴. نمودار مقایسه‌ی روند تغییرات خطای آموزش و آزمون هر سه مدل را طی دوره‌های آموزش رسم کنید.

۵. وزن‌های از‌پیش‌آموزش‌دیده‌ی VGG-16 اصلی را از torchvision بارگذاری کنید و آن را برای دسته‌بندی CIFAR-10 Fine-tune کنید.

---

در فصل ۱۲ با **معماری‌های مدرن CNN** آشنا می‌شویم؛ به‌خصوص ResNet که با ایده‌ی پیوند میان‌بر (Skip Connection) مشکل بزرگ گرادیان محوشونده در شبکه‌های خیلی عمیق را حل کرد و راه را برای آموزش شبکه‌هایی با صدها لایه گشود.
