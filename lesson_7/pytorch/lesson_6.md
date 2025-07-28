

## 🔹 صفحه ۶ — توابع مقایسه‌ای و شرطی در PyTorch

📌 شامل: `where()`, `gt()`, `lt()`, `eq()`, `clamp()`

---

### 🎯 چرا توابع شرطی و مقایسه‌ای اهمیت دارند؟

در بسیاری از مواقع نیاز داریم:

* مقادیر خاصی را فیلتر یا جایگزین کنیم
* فقط عناصر بزرگ‌تر یا کوچک‌تر از حد مشخصی را انتخاب کنیم
* داده‌ها را به محدوده خاصی محدود کنیم (مثلاً بین ۰ و ۱)

PyTorch مجموعه‌ای از توابع ساده برای انجام این کارها در اختیار ما می‌گذارد.

---

## 🔸 مقایسه عنصر به عنصر — `gt`, `lt`, `eq`

```python
import torch

x = torch.tensor([1, 4, 7, 10])

print(x.gt(5))   # Greater than 5
print(x.lt(5))   # Less than 5
print(x.eq(4))   # Equal to 4
```

خروجی‌ها به‌صورت Tensor با مقادیر Boolean خواهند بود:

```
tensor([False, False,  True,  True])
tensor([ True,  True, False, False])
tensor([False,  True, False, False])
```

---

## 🔸 استفاده از `where()` برای شرطی‌سازی

```python
x = torch.tensor([10, 20, 30, 40])
mask = x > 25

# Replace values: if condition True -> 1, else -> 0
result = torch.where(mask, torch.tensor(1), torch.tensor(0))

print(result)
```

خروجی:

```
tensor([0, 0, 1, 1])
```

---

## 🔸 محدودسازی مقادیر با `clamp()`

گاهی می‌خواهید مقادیر را بین دو حد نگه دارید—مثلاً هیچ مقداری کمتر از ۰ یا بیشتر از ۱ نباشد.

```python
x = torch.tensor([-2.0, 0.5, 1.5, 3.0])

# Clamp values between 0 and 1
clamped = x.clamp(min=0.0, max=1.0)

print(clamped)
```

خروجی:

```
tensor([0.0000, 0.5000, 1.0000, 1.0000])
```

---

## 🔸 مثال ترکیبی از مقایسه و where

```python
x = torch.tensor([4, 6, 8, 10])

# Replace all values greater than 6 with 99
x2 = torch.where(x > 6, torch.tensor(99), x)

print(x2)
```

خروجی:

```
tensor([ 4,  6, 99, 99])
```

---

## 🔍 نکته: توابع مقایسه‌ای معمولاً با ماسک‌ها (Boolean tensors) ترکیب می‌شوند تا عملیات شرطی دقیق انجام شود.

---

## 📘 تمرین پیشنهادی:

۱. یک Tensor ۱×۶ بسازید و با `clamp()` همه مقادیر را بین ۱۰ تا ۲۰ محدود کنید.
۲. یک لیست از نمرات بسازید و با `where()`، نمراتی که بالای ۱۲ هستند را به "قبول" تبدیل کنید و بقیه را "رد".
۳. بررسی کنید کدام مقادیر یک Tensor با مقدار خاص برابرند (`eq`) و سپس آن‌ها را با مقدار جدید جایگزین کنید.

---

