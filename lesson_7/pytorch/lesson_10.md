
## 🔹 صفحه ۱۰ — تبدیل نوع داده‌ها (Data Type Casting) در PyTorch

📌 شامل: `.float()`, `.int()`, `.long()`, `.bool()`, `.type()`

---

### 🎯 چرا تبدیل نوع داده اهمیت دارد؟

در PyTorch، عملیات ریاضی و منطقی باید روی Tensorهایی با نوع داده‌ی (dtype) یکسان انجام شود.
مثلاً ضرب یک Tensor `float` در یک Tensor `int` ممکن است خطا یا نتیجه‌ی اشتباه بدهد.
بنابراین باید بتوانیم به‌راحتی نوع داده را تغییر دهیم.

---

## 🔸 تبدیل به Float — `.float()`

```python
import torch

x = torch.tensor([1, 2, 3])
xf = x.float()

print(xf)
print(xf.dtype)
```

خروجی:

```
tensor([1., 2., 3.])
torch.float32
```

---

## 🔸 تبدیل به Integer — `.int()`, `.long()`, `.short()`

```python
x = torch.tensor([1.5, 2.7, 3.1])

print(x.int())   # torch.int32
print(x.long())  # torch.int64
```

خروجی:

```
tensor([1, 2, 3])
tensor([1, 2, 3])
```

✅ مقادیر اعشاری در این تبدیل **رُند نمی‌شوند**، بلکه فقط بخش صحیح نگه داشته می‌شود.

---

## 🔸 تبدیل به Boolean — `.bool()`

```python
x = torch.tensor([0, 1, 2, 0])

xb = x.bool()

print(xb)
```

خروجی:

```
tensor([False,  True,  True, False])
```

---

## 🔸 بررسی نوع دقیق با `.type()`

```python
x = torch.tensor([1.0])

print(x.type())   # torch.FloatTensor
```

✅ این متد برای بررسی دقیق نوع Tensor مفید است، خصوصاً در هنگام اشکال‌زدایی (debug).

---

## 🔸 تبدیل نوع داده با `.to(dtype=...)`

روش دیگری برای تغییر نوع داده با استفاده از `.to()` است:

```python
x = torch.tensor([1, 2, 3])
xf = x.to(dtype=torch.float32)

print(xf)
```

---

## 🔍 مقایسه مهم‌ترین نوع‌های داده در PyTorch:

| متد        | نوع داده      | مثال خروجی             |
| ---------- | ------------- | ---------------------- |
| `.float()` | torch.float32 | tensor(\[1., 2.])      |
| `.int()`   | torch.int32   | tensor(\[1, 2])        |
| `.long()`  | torch.int64   | tensor(\[1, 2])        |
| `.bool()`  | torch.bool    | tensor(\[True, False]) |

---

## 📘 تمرین پیشنهادی:

۱. یک Tensor از اعداد اعشاری بسازید و آن را به `.int()` و `.long()` تبدیل کنید.
۲. Tensor شامل صفر و عدد غیر صفر بسازید و با `.bool()` بررسی کنید کدام‌ها True می‌شوند.
۳. با استفاده از `.type()` نوع دقیق Tensorها را بررسی و چاپ کنید.

---
