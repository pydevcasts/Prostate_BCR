## 🔹 صفحه ۱۷ — آشنایی با تابع خطا (Loss Function) در PyTorch

📌 شامل: `nn.MSELoss()`, `nn.CrossEntropyLoss()`, نحوه استفاده و تفاوت‌ها

---

### 🎯 چرا تابع خطا اهمیت دارد؟

**تابع خطا (Loss Function)** معیار اصلی برای تعیین "میزان اشتباه" مدل است.
در هر مرحله از آموزش، مقدار خروجی مدل با مقدار واقعی مقایسه شده و تفاوت آن به‌صورت عددی اندازه‌گیری می‌شود.

✅ هرچه این مقدار کمتر باشد، یعنی مدل در حال یادگیری بهتر است.

---

## 🔸 تابع `nn.MSELoss()` — برای مسائل رگرسیون

این تابع اختلاف بین مقدار پیش‌بینی‌شده و مقدار واقعی را با استفاده از **میانگین مربعات خطا** محاسبه می‌کند.

```python
import torch
import torch.nn as nn

loss_fn = nn.MSELoss()

y_pred = torch.tensor([2.5, 0.0, 2.0])
y_true = torch.tensor([3.0, -0.5, 2.0])

loss = loss_fn(y_pred, y_true)

print(loss.item())
```

خروجی:

```
0.2917
```

✅ کاربرد اصلی این تابع در **مسائل رگرسیون** مثل پیش‌بینی قیمت، دما و غیره است.

---

## 🔸 تابع `nn.CrossEntropyLoss()` — برای طبقه‌بندی چندکلاسه

این تابع ترکیبی از **Softmax + Negative Log Likelihood** است.
نیاز به:

* خروجی خام مدل (بدون softmax)
* و لیبل‌های عددی صحیح (مثلاً ۰، ۱، ۲...)

```python
loss_fn = nn.CrossEntropyLoss()

# Predicted logits for 3 samples, each with 3 classes
y_pred = torch.tensor([[2.0, 1.0, 0.1],
                       [0.5, 2.5, 0.3],
                       [0.2, 0.3, 2.0]])

# True class indices
y_true = torch.tensor([0, 1, 2])

loss = loss_fn(y_pred, y_true)

print(loss.item())
```

خروجی:

```
1.0023
```

✅ کاربرد اصلی این تابع در **طبقه‌بندی چندکلاسه** مانند تشخیص دست‌خط، دسته‌بندی متن، تشخیص اشیا و غیره است.

---

## 🔸 تفاوت در ورودی‌ها

| تابع               | ورودی مدل                 | خروجی واقعی                |
| ------------------ | ------------------------- | -------------------------- |
| `MSELoss`          | مقدار پیوسته (float)      | مقدار پیوسته (float)       |
| `CrossEntropyLoss` | Logits خام (بدون softmax) | عدد صحیح (int: 0, 1, 2...) |

---

## 🔸 نحوه استفاده در مدل واقعی

در زمان آموزش یک مدل، تابع Loss در هر `forward pass` صدا زده می‌شود:

```python
model = nn.Sequential(
    nn.Linear(10, 3)  # 3-class classification
)

loss_fn = nn.CrossEntropyLoss()

x = torch.rand(4, 10)
y = torch.tensor([0, 2, 1, 0])

output = model(x)

loss = loss_fn(output, y)
loss.backward()
```

✅ بعد از محاسبه Loss، با `.backward()` گرادیان‌ها محاسبه می‌شوند.

---

## 🔍 لیست برخی توابع Loss رایج دیگر

| تابع                     | کاربرد                                     |
| ------------------------ | ------------------------------------------ |
| `nn.BCELoss()`           | طبقه‌بندی دودویی (binary) با خروجی Sigmoid |
| `nn.BCEWithLogitsLoss()` | ترکیب Sigmoid + BCE (بهتر)                 |
| `nn.L1Loss()`            | اختلاف مطلق، مقاوم در برابر داده پرت       |
| `nn.SmoothL1Loss()`      | ترکیب L1 و L2 برای پایداری بیشتر           |

---

## 📘 تمرین پیشنهادی:

۱. یک مدل با خروجی ۱ عدد بسازید و `MSELoss` برای آن تست کنید.
۲. یک مدل با خروجی چندکلاسه بسازید (مثلاً ۳ کلاس) و `CrossEntropyLoss` روی آن تست بزنید.
۳. فرق بین خروجی مدل قبل و بعد از softmax را مشاهده کنید.

---

✅ در صفحه بعدی وارد مبحث بسیار مهم **Optimizer ها** می‌شویم
📌 شامل: `SGD`, `Adam`, تنظیم نرخ یادگیری و اجرای `optimizer.step()`

---

