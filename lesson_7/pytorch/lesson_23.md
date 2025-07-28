## 🔹 صفحه ۲۳ — مدیریت batch، تقسیم داده و آماده‌سازی کامل برای آموزش

📌 شامل: `train_test_split`, `Subset`, تنظیم batch size، و استفاده از shuffle

---

### 🎯 چرا این موارد مهم‌اند؟

هنگام آموزش یک مدل حرفه‌ای، باید داده‌ها را به درستی:

* **تقسیم** کنیم به آموزش و آزمون (train/test)
* در قالب **batch** بخوانیم برای پردازش سریع‌تر
* به‌صورت تصادفی (shuffle) وارد کنیم تا مدل دچار وابستگی به ترتیب نشود

---

## 🔸 روش ۱: استفاده از `random_split` برای تقسیم train/test

```python
from torch.utils.data import random_split

dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = random_split(dataset, [train_size, test_size])
```

✅ حالا ۸۰٪ داده برای آموزش و ۲۰٪ برای تست استفاده خواهد شد.

---

## 🔸 ساخت DataLoader برای هر بخش

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
```

| پارامتر         | توضیح                        |
| --------------- | ---------------------------- |
| `batch_size=64` | تعداد نمونه در هر بار پردازش |
| `shuffle=True`  | ترتیب تصادفی برای آموزش بهتر |
| `shuffle=False` | در تست به ترتیب مهم نیست     |

---

## 🔸 مشاهده یک batch

```python
for images, labels in train_loader:
    print(images.shape)   # torch.Size([64, 1, 28, 28])
    print(labels[:5])     # First 5 labels
    break
```

---

## 🔸 روش ۲: استفاده از Subset (مثلاً برای استفاده محدود از داده)

```python
from torch.utils.data import Subset

indices = list(range(0, 1000))  # استفاده از ۱۰۰۰ نمونه اول فقط

subset = Subset(dataset, indices)

subset_loader = DataLoader(subset, batch_size=32, shuffle=True)
```

✅ گاهی برای تست سریع یا آموزش سریع‌تر، فقط روی بخشی از داده تمرین می‌کنیم.

---

## 🔍 نکات مهم برای آماده‌سازی داده:

| مورد             | توضیح                                       |
| ---------------- | ------------------------------------------- |
| Train/Test Split | برای ارزیابی عملکرد واقعی مدل               |
| Batch Size       | معمولاً بین 16 تا 128 انتخاب می‌شود         |
| Shuffle          | به جلوگیری از یادگیری ترتیب داده کمک می‌کند |
| Subset           | مفید برای تست یا مدل‌های سبک                |

---

## 📘 تمرین پیشنهادی:

۱. از مجموعه MNIST یک تقسیم ۷۰/۳۰ بسازید
۲. با batch size برابر 32 آموزش را تست کنید
۳. یک Subset با ۵۰۰ نمونه بسازید و روی آن مدل را تستی اجرا کنید

---

✅ در صفحه بعدی وارد فصل چهارم خواهیم شد:
**تبدیل داده به فرمت مناسب برای GPU و پردازش سریع‌تر**
📌 شامل: `.to(device)`, انتقال بین CPU و CUDA, و بررسی سرعت

---

