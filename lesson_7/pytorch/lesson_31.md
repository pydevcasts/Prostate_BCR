
# ✅ صفحه ۳۱ — تکنیک‌های پیشرفته در PyTorch برای بهبود عملکرد مدل

## 🔹 شامل: `Dropout`, `BatchNorm`, `Data Augmentation`, `Softmax`

📌 این تکنیک‌ها باعث افزایش دقت، کاهش overfitting و پایداری بیشتر مدل می‌شوند.

---

## 🔸 ۱. Dropout — حذف تصادفی نورون‌ها در حین آموزش

Dropout یک روش برای کاهش **overfitting** است. در هر مرحله آموزش، برخی نورون‌ها به‌طور تصادفی **غیرفعال** می‌شوند.

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),   # 50% of neurons randomly dropped
    nn.Linear(256, 10)
)
```

✅ فقط در **train mode** فعال است
✅ در `.eval()` خودکار غیرفعال می‌شود

---

## 🔸 ۲. Batch Normalization — نرمال‌سازی فعال‌سازی‌ها

BatchNorm باعث می‌شود:

* سرعت آموزش افزایش یابد
* شیب‌ها پایدارتر شوند
* نیاز به تنظیم دقیق learning rate کمتر شود

```python
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2)
)
```

✅ برای داده 2D مثل تصویر: `BatchNorm2d`
✅ برای لایه‌های خطی: `BatchNorm1d`

---

## 🔸 ۳. Data Augmentation — افزایش تنوع داده آموزشی

در داده‌های تصویری، می‌توان با تغییر شکل، چرخش، برش و روشنایی، داده‌های جدید ساخت.

```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(28, padding=2),
    transforms.ToTensor()
])
```

✅ با این روش، مدل با داده‌های متنوع‌تری آموزش می‌بیند
✅ فقط روی **train dataset** اعمال می‌شود

---

## 🔸 ۴. Softmax — تبدیل خروجی مدل به احتمال

برای مسائل طبقه‌بندی چندکلاسه (مانند MNIST)، از `Softmax` برای تبدیل خروجی به احتمال استفاده می‌کنیم.

```python
import torch.nn.functional as F

output = model(x)  # shape: [batch, 10]
probs = F.softmax(output, dim=1)

print(probs[0])  # Probabilities of 10 classes
```

✅ مجموع تمام خروجی‌ها برابر 1 می‌شود
✅ برای محاسبه loss نیازی نیست دستی اعمال شود، چون `CrossEntropyLoss` خودش آن را دارد

---

## 📘 تمرین نهایی:

۱. Dropout را با مقدارهای ۰.۳ و ۰.۵ امتحان کنید و دقت نهایی مدل را مقایسه کنید
۲. یک شبکه CNN با BatchNorm اضافه بسازید و سرعت همگرایی آن را بررسی کنید
۳. مدل را دو بار آموزش دهید: یک‌بار با Augmentation و یک‌بار بدون آن — تفاوت را بررسی کنید
۴. خروجی مدل خود را با `Softmax` تبدیل به احتمال کنید و کلاس پیش‌بینی‌شده را با `argmax` استخراج کنید

---

✅ با این صفحه، تکنیک‌های کلیدی برای بهبود عملکرد مدل را شناختید.
📌 این صفحه می‌تواند به‌عنوان ضمیمه یا فصل جمع‌بندی تکنیک‌های پیشرفته در پایان کتاب استفاده شود.

---

