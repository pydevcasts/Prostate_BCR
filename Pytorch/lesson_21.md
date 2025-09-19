

# 🟢 فصل سوم: کار با داده‌ها در PyTorch

## 🔹 صفحه ۲۱ — ساخت Dataset سفارشی با PyTorch

---

### 🎯 چرا Dataset مهم است؟

در PyTorch، داده‌ها از طریق کلاس `Dataset` مدیریت می‌شوند.
این ساختار اجازه می‌دهد که به‌راحتی:

* داده‌ها را **بارگذاری، مرتب‌سازی، تبدیل و برش** دهید
* از آن‌ها در **DataLoader** برای تقسیم به batch استفاده کنید
* آموزش مدل را ساختاریافته‌تر و مقیاس‌پذیرتر کنید

---

## 🔸 استفاده از `torch.utils.data.Dataset`

کلاس `Dataset` یک کلاس پایه است که باید آن را ارث‌بری کنیم و متدهای زیر را بازنویسی کنیم:

1. `__init__()` → بارگذاری داده یا تنظیم اولیه
2. `__getitem__()` → بازگرداندن یک نمونه با ایندکس خاص
3. `__len__()` → تعداد کل نمونه‌ها

---

## 🔸 مثال: ساخت Dataset فرضی

```python
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        self.x = torch.rand(100, 10)  # 100 samples, 10 features
        self.y = torch.rand(100, 1)   # 100 targets

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
```

✅ این کلاس حالا یک Dataset کاملاً قابل استفاده است.

---

## 🔸 استفاده از Dataset با DataLoader

```python
from torch.utils.data import DataLoader

dataset = MyDataset()
loader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch_x, batch_y in loader:
    print(batch_x.shape, batch_y.shape)
    break
```

خروجی:

```
torch.Size([16, 10]) torch.Size([16, 1])
```

✅ `shuffle=True` باعث می‌شود که داده‌ها در هر epoch به‌صورت تصادفی چیده شوند.

---

## 🔍 نکات مهم در طراحی Dataset

| مورد          | توضیح                                          |
| ------------- | ---------------------------------------------- |
| `__getitem__` | باید همیشه `(input, target)` بازگرداند         |
| `__len__`     | برای اطلاع از تعداد نمونه‌ها استفاده می‌شود    |
| داده‌ها       | می‌توانند tensor، numpy یا حتی آدرس فایل باشند |
| توابع تبدیل   | بعداً در `transform` می‌آیند (در صفحات بعدی)   |

---

## 📘 تمرین پیشنهادی:

۱. یک کلاس Dataset بسازید که ورودی آن فایل CSV باشد (مثلاً با `pandas.read_csv`)
۲. داده‌ها را به tensor تبدیل کنید و در `__getitem__` بازگردانید
۳. با `DataLoader` داده‌ها را batch بندی کنید و از آن‌ها در حلقه آموزش استفاده کنید

---

✅ در صفحه بعدی به سراغ **استفاده از توابع آماده Dataset مثل MNIST** و همچنین **نحوه تعریف transform روی داده‌ها** خواهیم رفت
📌 با استفاده از `torchvision.datasets` و `torchvision.transforms`

---
