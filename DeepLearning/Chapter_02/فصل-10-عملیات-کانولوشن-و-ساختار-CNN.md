# فصل ۱۰ 🔍 عملیات کانولوشن و ساختار کامل CNN

## 🎯 اهداف فصل

در فصل قبل دیدیم که MLP سه مشکل اساسی برای پردازش تصویر دارد: انفجار پارامترها، نادیده گرفتن ساختار فضایی و نبود ثبات موقعیتی. در این فصل یاد می‌گیریم که شبکه‌ی کانولوشنی (Convolutional Neural Network) چگونه هر سه این مشکل را با یک ایده‌ی ساده اما هوشمندانه حل می‌کند. در پایان این فصل خواهید توانست عملیات کانولوشن را با فرمول ریاضی و مثال عددی کامل توضیح دهید، مفاهیم فیلتر، Stride، Padding و نقشه‌ی ویژگی را درک کنید، لایه‌ی Pooling و انواع آن را بشناسید، معماری کامل یک CNN را از ورودی تا خروجی طراحی کنید، و یک CNN کامل را روی دیتاست واقعی CIFAR-10 آموزش دهید.

---

# 🔍 ایده‌ی اصلی کانولوشن: نگاه محلی به تصویر

تمام قدرت شبکه‌های کانولوشنی از یک ایده‌ی ساده ناشی می‌شود: به‌جای این‌که هر نورون به تمام پیکسل‌های تصویر متصل باشد (مثل MLP)، هر نورون تنها به یک ناحیه‌ی کوچک و محلی از تصویر نگاه می‌کند. این ناحیه‌ی کوچک را میدان پذیرنده (Receptive Field) می‌نامند. ابزاری که این نگاه محلی را ممکن می‌کند، یک ماتریس کوچک از اعداد به نام فیلتر (Filter) یا هسته (Kernel) است که روی تمام تصویر حرکت می‌کند و در هر موقعیت یک محاسبه انجام می‌دهد.

این رویکرد سه مشکل MLP را به‌یک‌باره حل می‌کند. اول، از آن‌جا که یک فیلتر روی تمام بخش‌های تصویر با همان وزن‌ها حرکت می‌کند (به این ویژگی اشتراک وزن می‌گویند)، تعداد پارامترها به‌شدت کاهش می‌یابد. دوم، از آن‌جا که فیلتر به ناحیه‌ی محلی نگاه می‌کند، روابط فضایی میان پیکسل‌های مجاور حفظ می‌شوند. سوم، از آن‌جا که همان فیلتر در تمام بخش‌های تصویر به‌کار می‌رود، اگر یک الگو (مثل لبه‌ی افقی) در گوشه‌ی چپ بالا باشد یا در مرکز تصویر، همان فیلتر آن را شناسایی می‌کند؛ این همان ثبات موقعیتی است که در MLP نداشتیم.

📷 [تصویر اینجا قرار گیرد: نمودار حرکت یک فیلتر ۳×۳ روی تصویر، با نمایش میدان پذیرنده در هر موقعیت و مقدار خروجی محاسبه‌شده]

---

# 📐 فرمول ریاضی کانولوشن دوبعدی

در حالت دوبعدی، خروجی کانولوشن در موقعیت (i, j) به شکل زیر محاسبه می‌شود؛ در این فرمول X تصویر ورودی، K فیلتر یا Kernel، و S خروجی کانولوشن است:

```
S(i,j) = sum_m sum_n [ X(i+m, j+n) * K(m,n) ]
```

به زبان ساده، در هر موقعیت، عناصر فیلتر در عناصر متناظر ناحیه‌ی محلی تصویر ضرب می‌شوند و مجموع این حاصل‌ضرب‌ها، مقدار خروجی در آن موقعیت است.

---

# 🔢 مثال عددی کامل کانولوشن

برای درک عمیق‌تر، یک مثال عددی گام‌به‌گام محاسبه می‌کنیم. فرض کنید بخشی از تصویر ورودی (ماتریس ۳×۳) به شکل زیر باشد و فیلتر نیز یک ماتریس ۳×۳ است:

تصویر ورودی:
```
1  2  3
4  5  6
7  8  9
```

فیلتر (برای تشخیص لبه‌های عمودی):
```
-1  0  1
-1  0  1
-1  0  1
```

برای محاسبه‌ی خروجی، هر عنصر فیلتر را در عنصر متناظر تصویر ضرب می‌کنیم: یک ضرب‌در منفی یک مساوی منفی یک، دو ضرب‌در صفر مساوی صفر، سه ضرب‌در یک مساوی سه، چهار ضرب‌در منفی یک مساوی منفی چهار، پنج ضرب‌در صفر مساوی صفر، شش ضرب‌در یک مساوی شش، هفت ضرب‌در منفی یک مساوی منفی هفت، هشت ضرب‌در صفر مساوی صفر، و نه ضرب‌در یک مساوی نه. مجموع نهایی برابر است با منفی یک به‌علاوه‌ی صفر به‌علاوه‌ی سه به‌علاوه‌ی منفی چهار به‌علاوه‌ی صفر به‌علاوه‌ی شش به‌علاوه‌ی منفی هفت به‌علاوه‌ی صفر به‌علاوه‌ی نه، که برابر با شش می‌شود. این فیلتر تفاوت میان سمت چپ و راست هر ناحیه را محاسبه می‌کند و به همین دلیل به‌عنوان تشخیص‌دهنده‌ی لبه‌ی عمودی عمل می‌کند.

---

# 🗺️ نقشه‌ی ویژگی (Feature Map)

خروجی یک عملیات کانولوشن با یک فیلتر، یک ماتریس دوبعدی به نام نقشه‌ی ویژگی (Feature Map) است. هر مقدار در این نقشه نشان می‌دهد که الگوی مورد نظر فیلتر در آن موقعیت از تصویر چقدر قوی است. یک لایه‌ی کانولوشنی معمولاً نه یک فیلتر، بلکه ده‌ها یا صدها فیلتر مختلف دارد که هرکدام در طول آموزش یاد می‌گیرد الگوی متفاوتی را تشخیص دهد. فیلترهای لایه‌های اولیه معمولاً الگوهای ساده‌ای مثل لبه‌ها و بافت‌ها یاد می‌گیرند، در حالی که فیلترهای لایه‌های عمیق‌تر، الگوهای پیچیده‌تری مثل اجزای اشیاء را تشخیص می‌دهند.

📷 [تصویر اینجا قرار گیرد: نمایش چند نقشه‌ی ویژگی حاصل از اعمال فیلترهای مختلف روی یک تصویر ورودی]

---

# 📏 Stride و تأثیر آن بر ابعاد خروجی

Stride تعداد پیکسل‌هایی است که فیلتر در هر حرکت جابه‌جا می‌شود. فرمول محاسبه‌ی ابعاد خروجی یک لایه‌ی کانولوشنی به شکل زیر است؛ در این فرمول H_in ارتفاع ورودی، K اندازه‌ی فیلتر، P مقدار Padding و S مقدار Stride است:

```
H_out = floor( (H_in + 2*P - K) / S ) + 1
```

برای مثال، اگر تصویر ورودی ۲۸ در ۲۸ باشد، فیلتر ۳ در ۳، Padding برابر صفر و Stride برابر یک باشد، ارتفاع خروجی برابر است با کف ((۲۸ + ۰ - ۳) / ۱) + ۱، که برابر با ۲۶ می‌شود.

---

# 📦 Padding؛ حفاظت از اطلاعات لبه‌ها

Padding با اضافه کردن یک حاشیه از مقادیر صفر دور تصویر، از کوچک‌شدن ابعاد خروجی جلوگیری می‌کند. با Padding برابر یک برای فیلتر ۳ در ۳، ابعاد خروجی برابر ابعاد ورودی می‌شود. مقدار مناسب Padding برای حفظ ابعاد با فیلتر K در K، برابر است با (K-1)/2.

📷 [تصویر اینجا قرار گیرد: مقایسه‌ی اعمال کانولوشن بدون Padding و با Padding=1]

---

# 🏊 لایه‌ی Pooling؛ کاهش هوشمند ابعاد

لایه‌ی Max Pooling در هر ناحیه‌ی کوچک (معمولاً ۲ در ۲) تنها بزرگ‌ترین مقدار را نگه می‌دارد. برای یک مثال عددی ساده با Max Pooling 2×2 و Stride=2:

```
ورودی ۴×۴:         خروجی ۲×۲:
1   3   2   4
5   6   1   2    →     6   4
7   8   3   2          8   5
4   3   5   1
```

ابعاد از ۴ در ۴ به ۲ در ۲ کاهش یافت. دلیل انتخاب بیشینه این است که مقدار بزرگ نشان‌دهنده‌ی حضور قوی یک ویژگی در آن ناحیه است.

📷 [تصویر اینجا قرار گیرد: نمودار اعمال Max Pooling 2×2 روی یک نقشه‌ی ویژگی با نمایش کادربندی ناحیه‌ها]

---

# 🏗️ معماری کامل CNN از ورودی تا خروجی

یک CNN استاندارد از دو بخش اصلی تشکیل می‌شود. بخش اول، استخراج‌کننده‌ی ویژگی (Feature Extractor) است که از تناوب لایه‌های کانولوشنی، Batch Normalization، تابع فعال‌سازی و Pooling تشکیل شده است. بخش دوم، طبقه‌بند (Classifier) است که از لایه‌های تمام‌متصل برای تبدیل نقشه‌های ویژگی به احتمالات کلاس‌ها تشکیل شده است. جریان کامل داده به این شکل است:

تصویر ورودی → کانولوشن + ReLU → Pooling → کانولوشن + ReLU → Pooling → Flatten → تمام‌متصل + ReLU → Softmax → احتمالات کلاس‌ها

📷 [تصویر اینجا قرار گیرد: نمودار کامل معماری یک CNN استاندارد با نمایش ابعاد داده در هر مرحله]

---

# 💻 پروژه‌ی عملی فصل: CNN کامل روی CIFAR-10

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
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


class CIFAR10_CNN(nn.Module):

    def __init__(self):
        super().__init__()

        # بلوک اول: ورودی (3,32,32) → خروجی (32,16,16)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )

        # بلوک دوم: ورودی (32,16,16) → خروجی (64,8,8)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )

        # بلوک سوم: ورودی (64,8,8) → خروجی (128,4,4)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )

        # طبقه‌بند: ورودی 128*4*4=2048 → خروجی 10
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"دستگاه محاسباتی: {device}")

model = CIFAR10_CNN().to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"تعداد پارامترها: {total_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

best_acc = 0.0

for epoch in range(30):

    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    train_acc = 100 * correct_train / total_train
    test_acc = 100 * correct_test / total_test

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_cifar10_cnn.pt')

    scheduler.step()

    if epoch % 5 == 0 or epoch == 29:
        print(f"دوره {epoch+1:2d} | خطا {running_loss/len(trainloader):.3f} | "
              f"آموزش {train_acc:.1f}% | آزمون {test_acc:.1f}% | بهترین {best_acc:.1f}%")

# ارزیابی دقیق به تفکیک هر کلاس
classes = ['هواپیما', 'خودرو', 'پرنده', 'گربه', 'آهو',
           'سگ', 'قورباغه', 'اسب', 'کشتی', 'کامیون']

model.load_state_dict(torch.load('best_cifar10_cnn.pt'))
model.eval()
class_correct = [0] * 10
class_total = [0] * 10

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(labels)):
            lbl = labels[i].item()
            class_correct[lbl] += (predicted[i] == labels[i]).item()
            class_total[lbl] += 1

print(f"\nدقت به تفکیک هر کلاس:")
for i in range(10):
    print(f"  {classes[i]:10s}: {100*class_correct[i]/class_total[i]:.1f}%")
```

در این پیاده‌سازی از دو لایه‌ی کانولوشنی متوالی پیش از هر Pooling استفاده شده که از معماری VGG الهام گرفته و به شبکه امکان می‌دهد ویژگی‌های پیچیده‌تری قبل از کاهش ابعاد یاد بگیرد. از CosineAnnealingLR به‌عنوان زمان‌بند نرخ یادگیری استفاده شده که نرخ یادگیری را به‌شکل موج کسینوسی کاهش می‌دهد و نتایج بهتری نسبت به کاهش خطی ساده نشان می‌دهد. در پایان آموزش نیز دقت مدل به تفکیک هر ده کلاس گزارش می‌شود؛ معمولاً تشخیص گربه و سگ به دلیل شباهت ظاهری، چالش‌برانگیزترین دسته‌ها هستند.

---

# ❓ سؤالات تستی

### سؤال ۱

اگر تصویر ورودی ۳۲ در ۳۲ باشد، فیلتر ۵ در ۵، Padding برابر ۲ و Stride برابر ۱، ابعاد خروجی چقدر خواهد بود؟

الف) ۲۸ در ۲۸

ب) ۳۰ در ۳۰

ج) ۳۲ در ۳۲

د) ۳۶ در ۳۶

### سؤال ۲

چرا در CNN وزن‌های یک فیلتر در تمام بخش‌های تصویر یکسان هستند (اشتراک وزن)؟

الف) چون این‌طور محاسبه ساده‌تر است

ب) چون اشتراک وزن باعث می‌شود همان الگو در هر موقعیتی از تصویر شناسایی شود و تعداد پارامترها به‌شدت کاهش یابد

ج) چون PyTorch از وزن‌های متفاوت پشتیبانی نمی‌کند

د) چون اشتراک وزن سرعت آموزش را کاهش می‌دهد

---

## ✅ پاسخ‌ها

**پاسخ سؤال ۱: گزینه ج**

با استفاده از فرمول: H_out = floor((32 + 2×2 - 5) / 1) + 1 = floor(31 / 1) + 1 = 32. ابعاد خروجی برابر ۳۲ در ۳۲ خواهد بود که همان ابعاد ورودی است.

**پاسخ سؤال ۲: گزینه ب**

اشتراک وزن دو مزیت اساسی دارد: تعداد پارامترها را از مقیاس تصویر مستقل می‌کند، و ثبات موقعیتی ایجاد می‌کند یعنی یک الگو هر جای تصویر باشد شناسایی می‌شود.

---

# 📝 خلاصه فصل

در این فصل با اصلی‌ترین عملیات شبکه‌های کانولوشنی آشنا شدیم. دیدیم که کانولوشن با فیلترهای کوچک متحرک هر سه محدودیت MLP را حل می‌کند. فرمول ریاضی و مثال عددی کانولوشن را بررسی کردیم. مفاهیم Stride، Padding و نقشه‌ی ویژگی را با فرمول و نمونه‌ی عددی فهمیدیم. Max Pooling را با مثال عددی درک کردیم و معماری کامل CNN را طراحی کردیم. در پروژه‌ی عملی یک CNN سه‌بلوکی کامل روی CIFAR-10 با Data Augmentation، Batch Normalization و زمان‌بند نرخ یادگیری پیاده‌سازی کردیم.

---

# 🎯 تمرین‌های پایان فصل

۱. ابعاد خروجی را برای تصویر ۶۴ در ۶۴، فیلتر ۷ در ۷، Padding برابر ۳ و Stride برابر ۲ محاسبه کنید.

۲. یک فیلتر ۳ در ۳ برای تشخیص لبه‌های افقی طراحی و روی یک تصویر ۵ در ۵ به‌صورت دستی اعمال کنید.

۳. Average Pooling را جایگزین Max Pooling کنید و تأثیر آن بر دقت را بسنجید.

۴. تعداد پارامترهای هر بلوک را به‌صورت دستی محاسبه کنید.

۵. اندازه‌ی فیلترها را در بلوک سوم از ۳ در ۳ به ۵ در ۵ تغییر دهید و تأثیر آن بر تعداد پارامترها و دقت را بررسی کنید.

---

در فصل ۱۱ با **معماری‌های کلاسیک CNN** آشنا می‌شویم؛ از LeNet که اولین CNN عملی تاریخ بود تا AlexNet که انقلاب یادگیری عمیق را کلید زد و VGG که نشان داد عمق شبکه کلید موفقیت است.
