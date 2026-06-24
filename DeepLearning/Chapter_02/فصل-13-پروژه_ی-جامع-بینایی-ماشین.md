# فصل ۱۳ 🎯 پروژه‌ی جامع بینایی ماشین: تشخیص بیماری پوستی با Transfer Learning

## 🎯 اهداف فصل

این فصل پروژه‌ی جامع بخش دوم کتاب است. تمام آنچه در فصل‌های ۹ تا ۱۲ آموختیم، یعنی پردازش تصویر، CNN، معماری‌های کلاسیک و مدرن، را در یک پروژه‌ی پزشکی واقعی و چالش‌برانگیز به کار می‌بریم. مسئله‌ای که انتخاب کرده‌ایم تشخیص بیماری پوستی از تصاویر درماتوسکوپی است؛ یعنی تصاویر ریزبینانه از پوست که پزشکان برای تشخیص سرطان پوست استفاده می‌کنند. در پایان این فصل خواهید توانست Transfer Learning را با مدل‌های از‌پیش‌آموزش‌دیده به‌صورت کامل پیاده‌سازی کنید، استراتژی‌های مختلف Fine-tuning را بشناسید و مقایسه کنید، با چالش‌های دنیای واقعی مثل عدم‌تعادل شدید کلاس‌ها و داده‌ی محدود دست‌وپنجه نرم کنید، و یک خط لوله‌ی (Pipeline) کامل بینایی ماشین صنعتی را از بارگذاری داده تا ذخیره و ارائه‌ی مدل پیاده‌سازی کنید.

---

# 🔬 تعریف مسئله: چرا این پروژه مهم است؟

سرطان پوست یکی از شایع‌ترین انواع سرطان در جهان است و تشخیص زودهنگام آن می‌تواند نرخ بقا را به‌طور چشمگیری افزایش دهد. اما تشخیص دقیق آن از روی تصویر، حتی برای متخصصان پوست هم چالش‌برانگیز است و نیاز به سال‌ها تجربه دارد. مدل‌های یادگیری عمیق در تحقیقات نشان داده‌اند که می‌توانند در تشخیص بعضی انواع سرطان پوست به دقتی نزدیک به متخصصان برسند. این پروژه از دیتاست HAM10000 استفاده می‌کند که شامل ده هزار تصویر درماتوسکوپی از هفت نوع ضایعه‌ی پوستی است و یکی از استانداردترین دیتاست‌ها در این حوزه به شمار می‌رود.

چالش‌های اصلی این پروژه که آن را از پروژه‌های قبلی متمایز می‌کند عبارتند از: عدم‌تعادل شدید کلاس‌ها (یک نوع ضایعه بیش از شصت و هفت درصد داده را تشکیل می‌دهد)، اندازه‌ی تصاویر که بزرگ‌تر از CIFAR-10 است، و اهمیت بالای Recall برای کلاس‌های خطرناک‌تر. برای مقابله با این چالش‌ها از Transfer Learning استفاده می‌کنیم که به جای آموزش از صفر، از دانش از‌پیش‌آموخته‌شده‌ی یک مدل بزرگ بهره می‌گیرد.

---

# 🔄 Transfer Learning چیست و چرا قدرتمند است؟

Transfer Learning یا یادگیری انتقالی، ایده‌ای است که در آن دانش یادگرفته‌شده برای حل یک مسئله را برای حل مسئله‌ی دیگری به کار می‌بریم. در یادگیری عمیق، این به این معناست که یک مدل که روی یک دیتاست بسیار بزرگ (مثل ImageNet با بیش از یک میلیون تصویر از هزار دسته) آموزش دیده را به‌عنوان نقطه‌ی شروع برای مسئله‌ی جدیدمان استفاده می‌کنیم.

دلیل کارآمدی این رویکرد این است که لایه‌های ابتدایی یک CNN از‌پیش‌آموزش‌دیده، ویژگی‌های پایه‌ای مثل لبه‌ها، بافت‌ها، و رنگ‌ها یاد گرفته‌اند که در تقریباً هر مسئله‌ی بینایی ماشین مفید هستند. حتی اگر دیتاست هدف ما (مثلاً تصاویر پوستی) بسیار متفاوت از ImageNet باشد، این ویژگی‌های پایه همچنان ارزشمند هستند و مدل نیازی ندارد آن‌ها را از صفر یاد بگیرد.

دو استراتژی اصلی برای Transfer Learning وجود دارد. استراتژی اول، Feature Extraction است که در آن لایه‌های کانولوشنی مدل از‌پیش‌آموزش‌دیده را کاملاً ثابت نگه می‌داریم (Freeze می‌کنیم) و فقط طبقه‌بند جدید را آموزش می‌دهیم. این روش سریع‌تر است و به داده‌ی کمتری نیاز دارد. استراتژی دوم، Fine-tuning است که در آن پس از چند دوره‌ی آموزش طبقه‌بند جدید، قفل لایه‌های کانولوشنی را هم باز می‌کنیم و اجازه می‌دهیم تمام شبکه با نرخ یادگیری بسیار کوچک‌تری به‌تدریج بر اساس داده‌ی جدید تنظیم شود. ترکیب این دو استراتژی به‌صورت پشت‌سرهم معمولاً بهترین نتیجه را می‌دهد.

📷 [تصویر اینجا قرار گیرد: نمودار مقایسه‌ی آموزش از صفر در برابر Transfer Learning با Feature Extraction و Fine-tuning، با نمایش اینکه کدام لایه‌ها در هر مرحله ثابت (آبی) یا قابل‌یادگیری (سبز) هستند]

---

# 🗂️ مرحله‌ی اول: بارگذاری و آماده‌سازی داده

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os

# ---- تعریف دیتاست سفارشی ----
# این کلاس داده‌ها را از یک CSV و پوشه‌ی تصاویر بارگذاری می‌کند
class SkinLesionDataset(Dataset):

    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.img_dir,
            self.df.loc[idx, 'image_id'] + '.jpg'
        )
        image = Image.open(img_name).convert('RGB')
        label = self.df.loc[idx, 'label']

        if self.transform:
            image = self.transform(image)

        return image, label


def prepare_data(metadata_path, img_dir, img_size=224):

    df = pd.read_csv(metadata_path)

    # نگاشت نام کلاس‌ها به اعداد
    class_names = sorted(df['dx'].unique())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    df['label'] = df['dx'].map(class_to_idx)

    # تقسیم داده با حفظ نسبت کلاس‌ها
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, random_state=42, stratify=train_df['label']
    )

    # تبدیل‌های Data Augmentation پیشرفته برای آموزش
    transform_train = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3,
            saturation=0.3, hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # برای اعتبارسنجی و آزمون فقط Resize و Normalize
    transform_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = SkinLesionDataset(train_df, img_dir, transform_train)
    val_dataset = SkinLesionDataset(val_df, img_dir, transform_eval)
    test_dataset = SkinLesionDataset(test_df, img_dir, transform_eval)

    # مدیریت عدم‌تعادل کلاس‌ها با WeightedRandomSampler
    # هر نمونه با احتمال وارون تعداد کلاسش نمونه‌برداری می‌شود
    class_counts = train_df['label'].value_counts().sort_index()
    class_weights = 1.0 / class_counts.values
    sample_weights = [
        class_weights[label] for label in train_df['label'].values
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    trainloader = DataLoader(
        train_dataset, batch_size=32, sampler=sampler, num_workers=2
    )
    valloader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=2
    )
    testloader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=2
    )

    print(f"آموزش: {len(train_dataset)} | اعتبارسنجی: {len(val_dataset)} | آزمون: {len(test_dataset)}")
    print(f"کلاس‌ها: {class_names}")
    print(f"توزیع کلاس در آموزش:\n{train_df['dx'].value_counts()}")

    return trainloader, valloader, testloader, class_names
```

---

# 🏗️ مرحله‌ی دوم: ساخت مدل با Transfer Learning

```python
def build_transfer_model(num_classes, pretrained=True):
    """
    ساخت مدل بر پایه ResNet-50 از‌پیش‌آموزش‌دیده روی ImageNet.
    در مرحله اول فقط طبقه‌بند جدید آموزش می‌بیند (Feature Extraction).
    """

    # بارگذاری ResNet-50 از‌پیش‌آموزش‌دیده
    model = models.resnet50(pretrained=pretrained)

    # مرحله اول: ثابت کردن تمام لایه‌های از‌پیش‌آموزش‌دیده
    for param in model.parameters():
        param.requires_grad = False

    # جایگزینی طبقه‌بند نهایی با طبقه‌بند جدید
    # fc اصلی ورودی ۲۰۴۸ نورون دارد
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    # شمارش پارامترهای قابل‌یادگیری در هر مرحله
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"پارامتر قابل‌یادگیری: {trainable:,} از {total:,} ({100*trainable/total:.1f}%)")

    return model


def unfreeze_for_finetuning(model, layers_to_unfreeze=2):
    """
    مرحله دوم: باز کردن قفل آخرین چند لایه برای Fine-tuning.
    این تابع لایه‌های layer4 و layer3 ResNet را قابل‌یادگیری می‌کند.
    """

    # ابتدا همه را ثابت نگه می‌داریم
    for param in model.parameters():
        param.requires_grad = False

    # سپس طبقه‌بند جدید را فعال می‌کنیم
    for param in model.fc.parameters():
        param.requires_grad = True

    # لایه‌های انتخابی ResNet را هم فعال می‌کنیم
    layers = [model.layer4, model.layer3, model.layer2, model.layer1]
    for i in range(min(layers_to_unfreeze, len(layers))):
        for param in layers[i].parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"پس از Fine-tuning — پارامتر قابل‌یادگیری: {trainable:,} ({100*trainable/total:.1f}%)")

    return model
```

---

# 🚀 مرحله‌ی سوم: آموزش دومرحله‌ای

```python
def train_epoch(model, loader, criterion, optimizer, device):

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient Clipping برای پایداری آموزش
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(loader), 100 * correct / total


def evaluate(model, loader, criterion, device):

    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return total_loss / len(loader), acc, all_preds, all_labels


def full_training_pipeline(
    trainloader, valloader, testloader,
    class_names, device, num_epochs_phase1=15, num_epochs_phase2=25
):

    num_classes = len(class_names)
    model = build_transfer_model(num_classes).to(device)

    # وزن‌دهی به تابع هزینه برای مدیریت عدم‌تعادل کلاس‌ها
    criterion = nn.CrossEntropyLoss()

    # ---- مرحله‌ی اول: Feature Extraction ----
    print("\n=== مرحله ۱: Feature Extraction ===")
    optimizer_phase1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )
    scheduler1 = optim.lr_scheduler.StepLR(
        optimizer_phase1, step_size=5, gamma=0.5
    )

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(num_epochs_phase1):
        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer_phase1, device
        )
        val_loss, val_acc, _, _ = evaluate(
            model, valloader, criterion, device
        )
        scheduler1.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"دوره {epoch+1:2d} | "
              f"آموزش: {train_acc:.1f}% | اعتبارسنجی: {val_acc:.1f}%")

    print(f"\nبهترین دقت اعتبارسنجی در مرحله ۱: {best_val_acc:.2f}%")

    # ---- مرحله‌ی دوم: Fine-tuning ----
    print("\n=== مرحله ۲: Fine-tuning (باز کردن قفل layer4 و layer3) ===")
    model.load_state_dict(best_weights)
    model = unfreeze_for_finetuning(model, layers_to_unfreeze=2)

    # نرخ یادگیری بسیار کوچک‌تر برای Fine-tuning
    # نرخ‌های متفاوت برای لایه‌های مختلف
    optimizer_phase2 = optim.AdamW([
        {'params': model.fc.parameters(), 'lr': 1e-4},
        {'params': model.layer4.parameters(), 'lr': 1e-5},
        {'params': model.layer3.parameters(), 'lr': 1e-5}
    ], weight_decay=1e-4)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phase2, T_max=num_epochs_phase2
    )

    best_val_acc_p2 = best_val_acc

    for epoch in range(num_epochs_phase2):
        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer_phase2, device
        )
        val_loss, val_acc, _, _ = evaluate(
            model, valloader, criterion, device
        )
        scheduler2.step()

        if val_acc > best_val_acc_p2:
            best_val_acc_p2 = val_acc
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_weights, 'best_skin_model.pt')

        if epoch % 5 == 0 or epoch == num_epochs_phase2 - 1:
            print(f"دوره {epoch+1:2d} | "
                  f"آموزش: {train_acc:.1f}% | "
                  f"اعتبارسنجی: {val_acc:.1f}% | "
                  f"بهترین: {best_val_acc_p2:.1f}%")

    # ---- ارزیابی نهایی ----
    print("\n=== ارزیابی نهایی روی مجموعه آزمون ===")
    model.load_state_dict(best_weights)
    _, test_acc, test_preds, test_labels = evaluate(
        model, testloader, criterion, device
    )

    print(f"\nدقت نهایی روی مجموعه آزمون: {test_acc:.2f}%")
    print(f"\nگزارش تفصیلی ارزیابی:\n")
    print(classification_report(
        test_labels, test_preds,
        target_names=class_names
    ))

    return model, test_acc


# اجرای کامل خط لوله
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"دستگاه محاسباتی: {device}")

# برای اجرای واقعی، مسیر دیتاست HAM10000 را وارد کنید
# trainloader, valloader, testloader, class_names = prepare_data(
#     metadata_path='HAM10000_metadata.csv',
#     img_dir='HAM10000_images/'
# )
# model, acc = full_training_pipeline(
#     trainloader, valloader, testloader,
#     class_names, device
# )

print("\nساختار کد آماده است. برای اجرا:")
print("1. دیتاست HAM10000 را از https://www.kaggle.com/datasets/kmader/skin-lesion-analysis دانلود کنید")
print("2. مسیر metadata_path و img_dir را در تابع prepare_data تنظیم کنید")
print("3. تابع full_training_pipeline را اجرا کنید")
```

---

# 📊 مرحله‌ی چهارم: تحلیل نتایج و تفسیر مدل

```python
def analyze_results(model, testloader, class_names, device):
    """
    تحلیل عمیق نتایج: یافتن کلاس‌هایی که مدل در آن‌ها ضعیف است
    و نمایش نمونه‌هایی که مدل اشتباه پیش‌بینی کرده
    """

    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # محاسبه دقت هر کلاس
    cm = confusion_matrix(all_labels, all_preds)
    print("\nماتریس درهم‌ریختگی:")
    print(cm)

    # شناسایی کلاس‌هایی با پایین‌ترین Recall
    print("\nRecall به تفکیک هر کلاس:")
    for i, cls in enumerate(class_names):
        if cm[i].sum() > 0:
            recall = cm[i, i] / cm[i].sum()
            print(f"  {cls:20s}: {recall*100:.1f}%")

    # اطمینان میانگین مدل برای پیش‌بینی‌های درست و غلط
    correct_confs = []
    wrong_confs = []
    for prob, pred, true in zip(all_probs, all_preds, all_labels):
        conf = prob[pred]
        if pred == true:
            correct_confs.append(conf)
        else:
            wrong_confs.append(conf)

    print(f"\nاطمینان میانگین برای پیش‌بینی‌های درست: {np.mean(correct_confs)*100:.1f}%")
    print(f"اطمینان میانگین برای پیش‌بینی‌های غلط: {np.mean(wrong_confs)*100:.1f}%")

    return all_preds, all_labels, all_probs


print("تابع تحلیل نتایج آماده است.")
```

---

# ❓ سؤالات تستی

### سؤال ۱

در مرحله‌ی Feature Extraction از Transfer Learning، چرا لایه‌های کانولوشنی مدل از‌پیش‌آموزش‌دیده ثابت نگه داشته می‌شوند؟

الف) چون لایه‌های کانولوشنی اصلاً در PyTorch قابل‌تغییر نیستند

ب) چون این لایه‌ها ویژگی‌های عمومی مفیدی یاد گرفته‌اند که می‌خواهیم حفظ شوند و با داده‌ی کم جدید خراب نشوند

ج) چون آموزش لایه‌های کانولوشنی سرعت را کاهش می‌دهد

د) چون لایه‌های ابتدایی هیچ اطلاعات مفیدی ندارند

### سؤال ۲

چرا در مرحله‌ی Fine-tuning از نرخ یادگیری بسیار کوچک‌تری (مثلاً ۱۰ برابر کوچک‌تر) نسبت به مرحله‌ی Feature Extraction استفاده می‌شود؟

الف) چون بهینه‌ساز در مرحله‌ی دوم باید کندتر کار کند

ب) چون وزن‌های از‌پیش‌آموزش‌دیده از قبل نزدیک به مقادیر بهینه هستند و تغییرات بزرگ می‌توانند دانش مفید قبلی را از بین ببرند

ج) چون PyTorch در مرحله‌ی Fine-tuning نرخ یادگیری پایین‌تر را الزامی می‌کند

د) چون داده‌ی آموزشی در مرحله‌ی دوم کمتر است

---

## ✅ پاسخ‌ها

**پاسخ سؤال ۱: گزینه ب**

لایه‌های کانولوشنی یک مدل از‌پیش‌آموزش‌دیده روی ImageNet، ویژگی‌های عمومی‌ای مثل لبه، بافت، رنگ و اشکال پایه را یاد گرفته‌اند که در تقریباً هر مسئله‌ی بینایی ماشین مفید هستند. اگر با داده‌ی کم آن‌ها را هم آموزش بدهیم، ممکن است این دانش عمومی از بین برود و دچار بیش‌برازش شدید شویم.

**پاسخ سؤال ۲: گزینه ب**

وزن‌های از‌پیش‌آموزش‌دیده پس از آموزش روی میلیون‌ها تصویر، به یک نقطه‌ی بسیار خوب در فضای پارامتر رسیده‌اند. نرخ یادگیری بزرگ می‌تواند این وزن‌ها را از این نقطه‌ی خوب دور کند و دانش ارزشمند قبلی را نابود کند. نرخ یادگیری کوچک‌تر اجازه می‌دهد وزن‌ها تنها کمی برای تطابق با دیتاست جدید تنظیم شوند.

---

# 📝 خلاصه فصل

در این فصل یک پروژه‌ی کامل بینایی ماشین در حوزه‌ی پزشکی پیاده‌سازی کردیم. دیتاست سفارشی پوستی را با Dataset سفارشی و WeightedRandomSampler برای مدیریت عدم‌تعادل کلاس‌ها آماده کردیم. مدل ResNet-50 از‌پیش‌آموزش‌دیده را با دو استراتژی Transfer Learning (Feature Extraction و Fine-tuning) آموزش دادیم و دیدیم که چرا Fine-tuning تدریجی با نرخ‌های یادگیری متفاوت برای لایه‌های مختلف، بهترین نتیجه را می‌دهد. همچنین با تحلیل نتایج به تفکیک کلاس و بررسی اطمینان مدل، درک عمیق‌تری از نقاط قوت و ضعف مدل پیدا کردیم. این پروژه نمونه‌ای واقعی از یک خط لوله‌ی صنعتی یادگیری عمیق برای بینایی ماشین است.

---

# 🎯 تمرین‌های پایان فصل

۱. به‌جای ResNet-50، از EfficientNet-B3 از کتابخانه‌ی timm استفاده کنید و نتایج را مقایسه کنید.

۲. یک رویکرد Test-Time Augmentation پیاده‌سازی کنید که هر تصویر آزمون را چندین بار با تبدیل‌های مختلف به مدل بدهد و میانگین پیش‌بینی‌ها را گزارش کند.

۳. از Grad-CAM برای تجسم مناطقی از تصویر که مدل بیشترین توجه را به آن‌ها دارد استفاده کنید.

۴. تأثیر WeightedRandomSampler را با آموزش مدل بدون آن مقایسه کنید و ببینید Recall کلاس اقلیت چقدر تغییر می‌کند.

۵. تابع `full_training_pipeline` را اصلاح کنید تا سه لایه (layer4، layer3 و layer2) را در مرحله‌ی Fine-tuning باز کند و نتیجه را با باز کردن دو لایه مقایسه کنید.

---

با پایان این فصل، بخش دوم کتاب که به بینایی ماشین اختصاص داشت به پایان رسید. در **بخش سوم** که از فصل ۱۴ آغاز می‌شود، وارد دنیای داده‌های ترتیبی و شبکه‌های بازگشتی می‌شویم؛ معماری‌هایی که برای یادگیری از داده‌هایی مثل متن، صدا و سری زمانی طراحی شده‌اند.
