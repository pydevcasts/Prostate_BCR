
## 🔹 صفحه ۹ — بررسی ویژگی‌های Tensor در PyTorch

📌 شامل: `.shape`, `.size()`, `.ndim`, `.dim()`, `.numel()`, `.dtype`, `.device`

---

### 🎯 چرا دانستن ویژگی‌های Tensor اهمیت دارد؟

در هنگام طراحی مدل، اشکال‌زدایی (debug)، یا مشاهده خروجی‌ها، لازم است:

* بدانیم Tensor چند بعد دارد
* هر بعد چه اندازه‌ای دارد
* نوع داده‌ی آن چیست
* روی کدام دستگاه (CPU یا GPU) قرار دارد

PyTorch با چند متد ساده، این اطلاعات را در اختیار ما قرار می‌دهد.

---

## 🔸 `.shape` — شکل کلی Tensor

```python
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(x.shape)
```

خروجی:

```
torch.Size([2, 3])
```

✅ `.shape` یک آبجکت از نوع `torch.Size` است که دقیقاً اندازه‌ی هر بعد را مشخص می‌کند.

---

## 🔸 `.size()` — مشابه `.shape()` ولی به شکل تابع

```python
print(x.size())
```

خروجی مشابه:

```
torch.Size([2, 3])
```

---

## 🔸 `.ndim` یا `.dim()` — تعداد ابعاد Tensor

```python
print(x.ndim)      # Property
print(x.dim())     # Method
```

خروجی:

```
2
2
```

---

## 🔸 `.numel()` — تعداد کل عناصر موجود در Tensor

```python
print(x.numel())
```

خروجی:

```
6
```

یعنی ۲ × ۳ = ۶ عنصر در این Tensor وجود دارد.

---

## 🔸 `.dtype` — نوع داده

```python
x1 = torch.tensor([1, 2, 3])
x2 = torch.tensor([1.0, 2.0, 3.0])

print(x1.dtype)    # torch.int64
print(x2.dtype)    # torch.float32
```

✅ تعیین نوع داده بسیار مهم است مخصوصاً هنگام محاسبات عددی، چون ترکیب نوع‌های مختلف ممکن است خطا ایجاد کند.

---

## 🔸 `.device` — مکان ذخیره Tensor (CPU یا GPU)

```python
x = torch.tensor([1.0, 2.0])

print(x.device)   # Shows 'cpu' by default

# Move to GPU if available
if torch.cuda.is_available():
    x = x.to("cuda")
    print(x.device)
```

---

## 🔍 جمع‌بندی سریع:

| ویژگی                | توضیح                    |
| -------------------- | ------------------------ |
| `.shape` / `.size()` | اندازه Tensor در هر بعد  |
| `.ndim` / `.dim()`   | تعداد ابعاد              |
| `.numel()`           | کل عناصر موجود           |
| `.dtype`             | نوع داده                 |
| `.device`            | محل نگهداری (CPU یا GPU) |

---

## 📘 تمرین پیشنهادی:

۱. یک Tensor با ابعاد `[3, 4, 2]` بسازید و ویژگی‌های `.shape`, `.ndim`, `.numel` را چاپ کنید.
۲. نوع داده یک Tensor را به `float32` تغییر دهید و `.dtype` را بررسی کنید.
۳. بررسی کنید Tensor روی کدام دستگاه است و در صورت وجود GPU آن را منتقل کنید.

---

