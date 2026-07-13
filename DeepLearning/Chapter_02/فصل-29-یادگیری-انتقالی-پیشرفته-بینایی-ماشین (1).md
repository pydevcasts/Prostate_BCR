# فصل ۲۹ 🔄 یادگیری انتقالی پیشرفته در بینایی ماشین

## 🎯 اهداف فصل

در فصل ۱۳ (پروژه‌ی جامع بخش دوم) با مفهوم Transfer Learning برای تشخیص بیماری پوستی آشنا شدیم. در این فصل به‌عنوان آخرین فصل بخش پنجم، به‌طور عمیق‌تر و جامع‌تر به استراتژی‌های مختلف یادگیری انتقالی در بینایی ماشین می‌پردازیم. در پایان این فصل خواهید توانست تفاوت Feature Extraction، Fine-tuning جزئی و Fine-tuning کامل را با مثال پیاده‌سازی کنید، استراتژی Freeze تدریجی لایه‌ها را بشناسید و در پروژه به کار ببرید، بدانید برای هر سناریو (داده‌ی کم/زیاد، دامنه‌ی مشابه/متفاوت) چه رویکردی مناسب‌تر است، و یک مدل تشخیص تصویر صنعتی با چندین استراتژی مقایسه کنید.

---

# 🗺️ نقشه‌ی راه انتخاب استراتژی

یکی از مهم‌ترین تصمیماتی که در یک پروژه‌ی بینایی ماشین باید بگیرید، انتخاب استراتژی مناسب Transfer Learning است. این تصمیم به دو عامل اصلی بستگی دارد: حجم داده‌ی آموزشی در دسترس و میزان شباهت دامنه‌ی هدف به دامنه‌ی پیش‌آموزش (معمولاً ImageNet).

وقتی داده کم و دامنه مشابه ImageNet است، Feature Extraction (Freeze کردن کامل backbone) بهترین انتخاب است چون کافی است فقط یک طبقه‌بند ساده روی ویژگی‌های از‌پیش‌آموزش‌دیده آموزش دهیم. وقتی داده زیاد و دامنه مشابه است، Fine-tuning جزئی (Freeze کردن لایه‌های اولیه، Fine-tuning لایه‌های انتهایی) بهترین عملکرد را دارد. وقتی داده کم اما دامنه متفاوت است، وضعیت سخت‌تر است و باید بین Feature Extraction و Fine-tuning تدریجی (که با احتیاط انجام می‌شود) انتخاب کنیم. وقتی داده زیاد و دامنه متفاوت است، Fine-tuning کامل منطقی‌ترین رویکرد است.

📷 [تصویر اینجا قرار گیرد: جدول ۲×۲ که چهار سناریوی (داده کم/زیاد) × (دامنه مشابه/متفاوت) را نشان می‌دهد و برای هر سناریو استراتژی پیشنهادی را مشخص می‌کند]

---

# ❄️ استراتژی‌های Freeze کردن لایه‌ها

در PyTorch، Freeze کردن یک لایه به این معناست که `requires_grad` تمام پارامترهای آن را برابر False قرار دهیم. این کار باعث می‌شود در طول پس‌انتشار خطا، گرادیان آن لایه محاسبه نشود و وزن‌هایش تغییر نکند. یک رویکرد پیشرفته که نتایج بهتری از Freeze کامل یا Fine-tuning کامل دارد، Fine-tuning تدریجی است: ابتدا فقط طبقه‌بند نهایی را آموزش می‌دهیم، سپس به‌تدریج لایه‌های قبلی را یکی‌یکی باز می‌کنیم. این رویکرد از بین رفتن دانش از‌پیش‌آموزش‌دیده را کاهش می‌دهد.

---

# 💻 پروژه‌ی عملی: مقایسه‌ی استراتژی‌های Transfer Learning روی دیتاست گل‌های آکسفورد

در این پروژه از دیتاست Flowers102 آکسفورد استفاده می‌کنیم که شامل ۱۰۲ دسته‌ی گل با تعداد نمونه‌ی نسبتاً کم در هر دسته است. این دیتاست برای مقایسه‌ی استراتژی‌ها ایده‌آل است چون دامنه‌اش کمی با ImageNet متفاوت است و داده‌ی کافی برای Fine-tuning کامل ندارد.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"دستگاه محاسباتی: {device}")

# ---- آماده‌سازی داده ----
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# بارگذاری دیتاست Flowers102
try:
    full_dataset = torchvision.datasets.Flowers102(
        root='./data', split='train', download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.Flowers102(
        root='./data', split='test', download=True, transform=transform_val
    )
    NUM_CLASSES = 102
    print(f"دیتاست Flowers102 بارگذاری شد: {len(full_dataset)} نمونه آموزشی")
except Exception as e:
    print(f"خطا در بارگذاری Flowers102: {e}")
    print("استفاده از CIFAR-100 به‌عنوان جایگزین...")
    full_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_val
    )
    NUM_CLASSES = 100

# تقسیم به train و validation
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
valloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"آموزش: {train_size} | اعتبارسنجی: {val_size} | آزمون: {len(test_dataset)}")

# ---- توابع کمکی ----
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = True

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), 100 * correct / total

def train_model(model, trainloader, valloader, optimizer, criterion,
                scheduler, num_epochs, strategy_name):
    print(f"\n--- استراتژی: {strategy_name} ---")
    print(f"پارامترهای قابل‌یادگیری: {count_trainable_params(model):,}")

    best_val_acc = 0.0
    best_weights = None
    start = time.time()

    for epoch in range(num_epochs):
        model.train()
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

        val_loss, val_acc = evaluate(model, valloader, criterion)
        if scheduler:
            scheduler.step(val_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 3 == 0 or epoch == num_epochs - 1:
            print(f"  دوره {epoch+1:2d}/{num_epochs} | اعتبارسنجی: {val_acc:.1f}%")

    elapsed = time.time() - start
    model.load_state_dict(best_weights)
    _, test_acc = evaluate(model, testloader, criterion)

    return {
        'strategy': strategy_name,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'trainable_params': count_trainable_params(model),
        'time_sec': elapsed
    }

criterion = nn.CrossEntropyLoss()
NUM_EPOCHS = 10
results = []

# ======== استراتژی ۱: Feature Extraction (Backbone کاملاً Freeze) ========
model1 = models.resnet50(pretrained=True).to(device)
freeze_all(model1)
model1.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model1.fc.in_features, NUM_CLASSES)
).to(device)

optimizer1 = optim.Adam(model1.fc.parameters(), lr=0.001)
scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, patience=3, factor=0.5)

r1 = train_model(model1, trainloader, valloader, optimizer1,
                  criterion, scheduler1, NUM_EPOCHS, "Feature Extraction")
results.append(r1)

# ======== استراتژی ۲: Fine-tuning جزئی (فقط layer4 + fc باز) ========
model2 = models.resnet50(pretrained=True).to(device)
freeze_all(model2)
unfreeze_layer(model2.layer4)
model2.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model2.fc.in_features, NUM_CLASSES)
).to(device)

optimizer2 = optim.AdamW([
    {'params': model2.layer4.parameters(), 'lr': 1e-5},
    {'params': model2.fc.parameters(), 'lr': 1e-3}
], weight_decay=1e-4)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=NUM_EPOCHS)

r2 = train_model(model2, trainloader, valloader, optimizer2,
                  criterion, scheduler2, NUM_EPOCHS, "Fine-tuning جزئی (layer4)")
results.append(r2)

# ======== استراتژی ۳: Fine-tuning تدریجی ========
model3 = models.resnet50(pretrained=True).to(device)
freeze_all(model3)
model3.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model3.fc.in_features, NUM_CLASSES)
).to(device)

print("\n--- استراتژی: Fine-tuning تدریجی ---")

# مرحله ۱: فقط fc
optimizer3a = optim.Adam(model3.fc.parameters(), lr=1e-3)
for epoch in range(3):
    model3.train()
    for imgs, labels in trainloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer3a.zero_grad()
        criterion(model3(imgs), labels).backward()
        optimizer3a.step()
print("  مرحله ۱ (فقط fc): ۳ دوره کامل شد")

# مرحله ۲: layer4 + fc
unfreeze_layer(model3.layer4)
optimizer3b = optim.AdamW([
    {'params': model3.layer4.parameters(), 'lr': 5e-6},
    {'params': model3.fc.parameters(), 'lr': 5e-4}
], weight_decay=1e-4)
for epoch in range(4):
    model3.train()
    for imgs, labels in trainloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer3b.zero_grad()
        criterion(model3(imgs), labels).backward()
        optimizer3b.step()
print("  مرحله ۲ (layer4+fc): ۴ دوره کامل شد")

# مرحله ۳: layer3 + layer4 + fc
unfreeze_layer(model3.layer3)
optimizer3c = optim.AdamW([
    {'params': model3.layer3.parameters(), 'lr': 1e-6},
    {'params': model3.layer4.parameters(), 'lr': 5e-6},
    {'params': model3.fc.parameters(), 'lr': 1e-4}
], weight_decay=1e-4)
for epoch in range(3):
    model3.train()
    for imgs, labels in trainloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer3c.zero_grad()
        criterion(model3(imgs), labels).backward()
        optimizer3c.step()
print("  مرحله ۳ (layer3+layer4+fc): ۳ دوره کامل شد")

_, val_acc3 = evaluate(model3, valloader, criterion)
_, test_acc3 = evaluate(model3, testloader, criterion)
results.append({
    'strategy': 'Fine-tuning تدریجی',
    'best_val_acc': val_acc3,
    'test_acc': test_acc3,
    'trainable_params': count_trainable_params(model3),
    'time_sec': 0
})

# ======== نمایش نتایج ========
print("\n" + "="*70)
print(f"{'استراتژی':<25} {'val_acc':<12} {'test_acc':<12} {'پارامتر'}")
print("-"*70)
for r in results:
    print(f"{r['strategy']:<25} {r['best_val_acc']:<12.1f} "
          f"{r['test_acc']:<12.1f} {r['trainable_params']:,}")
print("="*70)

# ذخیره‌ی بهترین مدل
best_result = max(results, key=lambda x: x['test_acc'])
print(f"\nبهترین استراتژی: {best_result['strategy']}")
print(f"دقت آزمون: {best_result['test_acc']:.2f}%")
torch.save(model3.state_dict(), 'best_transfer_model.pt')
print("مدل بهترین ذخیره شد.")
```

---

# 📊 راهنمای انتخاب استراتژی

| سناریو | داده | دامنه | استراتژی پیشنهادی |
|---|---|---|---|
| تشخیص گل با چند صد تصویر | کم | مشابه | Feature Extraction |
| تشخیص ضایعه‌ی پوستی | متوسط | کمی متفاوت | Fine-tuning تدریجی |
| طبقه‌بندی تصاویر ماهواره‌ای | زیاد | خیلی متفاوت | Fine-tuning کامل |
| تشخیص محصول کارخانه | کم | متفاوت | Feature Extraction + Data Aug |

---

# ❓ سؤالات تستی

### سؤال ۱

در Fine-tuning تدریجی، چرا لایه‌های ابتدایی شبکه دیرتر از لایه‌های انتهایی باز می‌شوند؟

الف) چون لایه‌های ابتدایی پارامتر بیشتری دارند

ب) چون لایه‌های ابتدایی ویژگی‌های پایه‌ی عمومی‌تر (مثل لبه و بافت) یاد گرفته‌اند که در اکثر وظایف مفیدند و کمتر نیاز به تغییر دارند؛ لایه‌های انتهایی ویژگی‌های اختصاصی‌تری دارند که بیشتر نیاز به تنظیم برای وظیفه‌ی جدید دارند

ج) چون PyTorch لایه‌های ابتدایی را سریع‌تر باز می‌کند

د) چون لایه‌های انتهایی در طول پیش‌آموزش آموزش ندیده‌اند

### سؤال ۲

وقتی از Feature Extraction استفاده می‌کنیم و Backbone را Freeze می‌کنیم، چرا نرخ یادگیری طبقه‌بند جدید را بزرگ‌تر از حالت Fine-tuning انتخاب می‌کنیم؟

الف) چون طبقه‌بند جدید از صفر شروع می‌کند و نیاز به تغییرات بزرگ‌تر برای رسیدن به وزن‌های مناسب دارد، در حالی که در Fine-tuning وزن‌های Backbone از قبل خوب هستند

ب) چون Freeze کردن نرخ یادگیری را به‌طور خودکار افزایش می‌دهد

ج) چون طبقه‌بند جدید پارامتر کمتری دارد

د) چون Backbone در Feature Extraction اصلاً نیازی به نرخ یادگیری ندارد

---

## ✅ پاسخ‌ها

**پاسخ سؤال ۱: گزینه ب**

تحقیقات نشان داده که لایه‌های اولیه‌ی CNN‌های عمیق ویژگی‌های کاملاً عمومی مثل لبه‌ها، زوایا و بافت‌های ساده یاد می‌گیرند که تقریباً در هر وظیفه‌ی بینایی مفیدند. لایه‌های انتهایی ویژگی‌های اختصاصی‌تری مثل اجزای اشیاء دارند که نیاز بیشتری به تطبیق با وظیفه‌ی جدید دارند.

**پاسخ سؤال ۲: گزینه الف**

طبقه‌بند جدید با وزن‌های تصادفی شروع می‌کند و نیاز دارد به‌سرعت به مقادیر مفید برسد. نرخ یادگیری بزرگ‌تر (مثل ۰.۰۰۱) این سرعت را فراهم می‌کند. اما اگر در Fine-tuning هم همین نرخ را برای Backbone استفاده کنیم، تغییرات بزرگ وزن‌های از‌پیش‌آموزش‌دیده را خراب می‌کند.

---

# 📝 خلاصه فصل

در این فصل به‌صورت عمیق با استراتژی‌های مختلف یادگیری انتقالی در بینایی ماشین آشنا شدیم. نقشه‌ی راهی برای انتخاب استراتژی مناسب بر اساس حجم داده و شباهت دامنه ارائه دادیم. سه استراتژی اصلی Feature Extraction، Fine-tuning جزئی و Fine-tuning تدریجی را پیاده‌سازی و مقایسه کردیم. نشان دادیم که Fine-tuning تدریجی با آزاد کردن تدریجی لایه‌ها و استفاده از نرخ‌های یادگیری متفاوت برای لایه‌های مختلف، اغلب بهترین تعادل میان حفظ دانش از‌پیش‌آموزش‌دیده و تطبیق با وظیفه‌ی جدید را دارد.

---

# 🎯 تمرین‌های پایان فصل

۱. به‌جای ResNet-50، از EfficientNet-B3 استفاده کنید و استراتژی‌های مشابه را مقایسه کنید.
۲. تأثیر Data Augmentation قوی‌تر را روی نتایج Feature Extraction بررسی کنید.
۳. یک دیتاست پزشکی یا صنعتی واقعی پیدا کنید و بهترین استراتژی را برای آن تعیین کنید.
۴. منحنی یادگیری (Learning Curve) هر سه استراتژی را رسم و مقایسه کنید.
۵. Test-Time Augmentation را پیاده‌سازی کنید که چندین نسخه‌ی augmented از هر تصویر آزمون را به مدل می‌دهد و میانگین پیش‌بینی‌ها را برمی‌گرداند.

---

در فصل ۳۰ به **استقرار مدل (Deployment) و عملیاتی‌سازی** می‌پردازیم؛ آخرین مرحله‌ی چرخه‌ی حیات یک پروژه‌ی یادگیری عمیق که مدل را از محیط آزمایشگاهی به دنیای واقعی منتقل می‌کند.
