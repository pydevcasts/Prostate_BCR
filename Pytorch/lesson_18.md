## 🔹 صفحه ۱۸ — آموزش مدل و استفاده از Optimizer در PyTorch

📌 شامل: `torch.optim.SGD`, `torch.optim.Adam`, مفهوم `learning rate`, و نحوه‌ی اجرای `.zero_grad()`, `.step()`

---

### 🎯 چرا Optimizer مهم است؟

پس از محاسبه خطا (Loss)، باید پارامترهای مدل را طوری تنظیم کنیم که این خطا **کمتر شود**.
این کار با کمک **Optimizer** انجام می‌شود، که با استفاده از گرادیان‌ها مقدار پارامترها را به‌روز می‌کند.

PyTorch ابزارهای متنوعی برای این کار ارائه می‌دهد که در این صفحه با مهم‌ترین آن‌ها آشنا می‌شویم.

---

## 🔸 مراحل کلی آموزش در PyTorch:

1. عبور داده از مدل → خروجی
2. محاسبه Loss بین خروجی و مقدار واقعی
3. محاسبه گرادیان‌ها با `.backward()`
4. بروزرسانی پارامترها با `.step()`
5. پاک کردن گرادیان‌های قدیمی با `.zero_grad()`

---

## 🔸 استفاده از `torch.optim.SGD`

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

✅ `SGD` یا **Stochastic Gradient Descent** یکی از ساده‌ترین و قدیمی‌ترین الگوریتم‌های بهینه‌سازی است.
پارامتر `lr=0.01` همان **learning rate** است.

---

## 🔸 اجرای یک قدم از آموزش

```python
# Input and target
x = torch.rand(4, 10)
y = torch.rand(4, 1)

# Forward
output = model(x)

# Loss
loss_fn = nn.MSELoss()
loss = loss_fn(output, y)

# Backward
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

| مرحله         | توضیح                             |
| ------------- | --------------------------------- |
| `zero_grad()` | پاک کردن گرادیان‌های قبلی         |
| `backward()`  | محاسبه گرادیان جدید               |
| `step()`      | بروزرسانی پارامترها با گرادیان‌ها |

---

## 🔸 استفاده از `torch.optim.Adam`

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

✅ Adam ترکیبی از **SGD + Momentum + Adaptive Learning** است و برای اکثر پروژه‌ها انتخاب پیش‌فرض مناسبی است.

---

## 🔍 مقایسه سریع Optimizerها:

| Optimizer | سرعت همگرایی         | پارامترهای اضافه | مناسب برای                   |
| --------- | -------------------- | ---------------- | ---------------------------- |
| `SGD`     | کندتر ولی قابل کنترل | momentum, lr     | پروژه‌های ساده یا قابل تنظیم |
| `Adam`    | سریع‌تر              | betas, eps, lr   | بیشتر پروژه‌های مدرن         |

---

## 🔸 نحوه تغییر Learning Rate

```python
# کاهش تدریجی lr در طول آموزش
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

✅ این ابزارها برای کنترل داینامیک lr در پروژه‌های واقعی مفید هستند ولی برای شروع نیازی نیست.

---

## 📘 تمرین پیشنهادی:

۱. یک مدل ساده با `Linear(5,1)` بسازید و روی آن `SGD` و سپس `Adam` تست کنید.
۲. مقدار loss را در دو حلقه آموزشی با هر Optimizer مقایسه کنید.
۳. با تغییر مقدار `lr`، اثر آن را در سرعت کاهش loss بررسی کنید.

---


