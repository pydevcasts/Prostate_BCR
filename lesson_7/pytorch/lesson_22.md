## 🔹 صفحه ۲۲ — استفاده از Datasetهای آماده و اعمال `transform` روی داده‌ها

📌 شامل: `torchvision.datasets`, استفاده از MNIST, اعمال `transforms`

---

### 🎯 چرا از Datasetهای آماده استفاده کنیم؟

بسیاری از داده‌های پرکاربرد (مانند MNIST، CIFAR10، ImageNet) توسط PyTorch به‌صورت آماده فراهم شده‌اند.
با استفاده از آن‌ها دیگر نیازی به دانلود دستی یا بارگذاری سفارشی نیست.

---

## 🔸 نصب و وارد کردن `torchvision`

برای استفاده از Datasetهای تصویری، به `torchvision` نیاز دارید:

```bash
pip install torchvision
```

سپس در کد:

```python
import torchvision
import torchvision.transforms as transforms
```

---

## 🔸 نمونه: استفاده از مجموعه MNIST

```python
transform = transforms.ToTensor()

train_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_data = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
```

✅ داده‌ها به‌صورت خودکار دانلود و به `Tensor` تبدیل می‌شوند.

---

## 🔸 استفاده از `DataLoader`

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)
```

اکنون می‌توانید داده‌ها را به‌صورت batch و آماده برای مدل دریافت کنید:

```python
for images, labels in train_loader:
    print(images.shape)   # torch.Size([32, 1, 28, 28])
    print(labels.shape)   # torch.Size([32])
    break
```

---

## 🔸 معرفی `transforms` متنوع

Transform‌ها برای **تبدیل، نرمال‌سازی، چرخش، تغییر اندازه، تغییر روشنایی و غیره** استفاده می‌شوند.

| Transform                           | توضیح                 |
| ----------------------------------- | --------------------- |
| `transforms.ToTensor()`             | تبدیل تصویر به Tensor |
| `transforms.Normalize(mean, std)`   | نرمال‌سازی داده       |
| `transforms.Resize((h, w))`         | تغییر اندازه          |
| `transforms.RandomHorizontalFlip()` | وارونگی افقی          |
| `transforms.Compose([...])`         | ترکیب چند transform   |

---

## 🔸 ترکیب چند transform

```python
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

---

## 🔍 نکته: چرا Normalize می‌کنیم؟

برای اینکه مدل سریع‌تر و بهتر یاد بگیرد، باید ورودی‌ها را در محدوده‌ی مناسب (مثلاً بین -1 و 1) قرار دهیم.
در تصاویر سیاه‌وسفید MNIST:

```python
transforms.Normalize(mean=[0.5], std=[0.5])  # → داده بین [-1, 1]
```

---

## 📘 تمرین پیشنهادی:

۱. مجموعه MNIST را دانلود و به DataLoader تبدیل کنید.
۲. از Transform برای تغییر اندازه تصویر به `32x32` استفاده کنید.
۳. داده را نرمال‌سازی کنید و با `imshow` نمایش دهید.
۴. تفاوت تصاویر نرمال‌شده و نشده را بررسی کنید.

---

✅ در صفحه بعدی به بررسی **مدیریت batch، تقسیم داده، و آماده‌سازی کامل داده‌ها برای آموزش** خواهیم پرداخت
📌 شامل: `train_test_split`, نمونه‌سازی `Subset`, استفاده از `batch size`, و shuffle

---

