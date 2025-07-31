
## 🔹 صفحه ۸ — اتصال، ترکیب، و تقسیم Tensorها در PyTorch

📌 شامل: `cat()`, `stack()`, `split()`, `chunk()`

---

### 🎯 چرا این توابع مهم هستند؟

در کاربردهای مختلف یادگیری ماشین، لازم است:

* چند Tensor را کنار هم بچسبانیم (مثلاً داده‌های batch)
* یا برعکس، یک Tensor بزرگ را به قسمت‌های کوچکتر تقسیم کنیم
* یا به‌صورت مؤثر داده را ترکیب کنیم در بعد خاص

پای تورچ ابزارهای دقیقی برای این کارها دارد.

---

## 🔸 اتصال با `torch.cat()` — Concatenation

این تابع چند Tensor را **در یک بعد مشخص** به هم متصل می‌کند.

```python
import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# Concatenate along rows (dim=0)
result1 = torch.cat((a, b), dim=0)

# Concatenate along columns (dim=1)
result2 = torch.cat((a, b), dim=1)

print(result1)
print(result2)
```

خروجی:

```
tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]])

tensor([[1, 2, 5, 6],
        [3, 4, 7, 8]])
```

---

## 🔸 ترکیب با `torch.stack()` — Stacking

بر خلاف `cat()` که ابعاد ثابت نگه می‌دارد، تابع `stack()` **یک بُعد جدید اضافه می‌کند**.

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Stack along new dimension (default dim=0)
stacked = torch.stack((a, b), dim=0)

print(stacked)
print(stacked.shape)
```

خروجی:

```
tensor([[1, 2, 3],
        [4, 5, 6]])
torch.Size([2, 3])
```

اگر `dim=1` تنظیم شود، بُعد جدید در جای دیگر قرار می‌گیرد:

```
tensor([[1, 4],
        [2, 5],
        [3, 6]])
```

---

## 🔸 تقسیم با `torch.split()`

این تابع یک Tensor را به چند بخش با اندازه‌های مشخص تقسیم می‌کند.

```python
x = torch.tensor([10, 20, 30, 40, 50, 60])

# Split into chunks of size 2
parts = torch.split(x, 2)

for part in parts:
    print(part)
```

خروجی:

```
tensor([10, 20])
tensor([30, 40])
tensor([50, 60])
```

---

## 🔸 تقسیم با `torch.chunk()`

این تابع مشابه `split()` است، ولی شما تعداد قسمت‌ها را مشخص می‌کنید و PyTorch **خودش اندازه‌ها را محاسبه می‌کند**.

```python
x = torch.tensor([1, 2, 3, 4, 5])

# Chunk into 3 parts
chunks = torch.chunk(x, 3)

for c in chunks:
    print(c)
```

خروجی:

```
tensor([1, 2])
tensor([3, 4])
tensor([5])
```

---

## 🔍 تفاوت‌های کلیدی:

| تابع      | افزودن بُعد جدید؟ | کنترل ابعاد؟                        | مناسب برای؟  |
| --------- | ----------------- | ----------------------------------- | ------------ |
| `cat()`   | ❌                 | باید ابعاد موجود match باشند        | چسباندن ساده |
| `stack()` | ✅                 | ابعاد ورودی باید دقیقاً یکسان باشند | ساخت batch   |
| `split()` | ❌                 | شما اندازه هر بخش را تعیین می‌کنید  | تقسیم دلخواه |
| `chunk()` | ❌                 | شما فقط تعداد بخش را می‌دهید        | تقسیم سریع   |

---

## 📘 تمرین پیشنهادی:

۱. دو Tensor با شکل `[2, 3]` بسازید و آن‌ها را با `cat()` در هر دو محور بچسبانید.
۲. چند Tensor یک‌بعدی با `stack()` ترکیب کنید و تغییر شکل را بررسی کنید.
۳. یک لیست عددی طولانی بسازید و آن را با `split()` به قسمت‌های مساوی ۴ عددی ببرید.
۴. همان لیست را با `chunk()` به ۳ قسمت ببرید.

---
