
## 🔹 صفحه ۵ — توابع آماری پرکاربرد در PyTorch

---

### 🎯 چرا توابع آماری مهم هستند؟

در تحلیل داده و یادگیری ماشین، بسیار رایج است که بخواهیم ویژگی‌هایی مثل میانگین، بیشینه، و انحراف معیار را از داده‌ها استخراج کنیم. PyTorch با ارائه متدهای ساده روی Tensorها، این کار را بسیار سریع و بدون نیاز به NumPy انجام می‌دهد.

---

## 🔸 محاسبه‌ی میانگین — `mean()`

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Calculate mean of all elements
print(x.mean())

# Calculate mean along dimension 0 (column-wise)
print(x.mean(dim=0))

# Calculate mean along dimension 1 (row-wise)
print(x.mean(dim=1))
```

---

## 🔸 انحراف معیار — `std()`

```python
# Calculate standard deviation of all elements
print(x.std())

# Standard deviation along each column
print(x.std(dim=0))
```

---

## 🔸 مجموع عناصر — `sum()`

```python
print(x.sum())         # Sum of all elements
print(x.sum(dim=1))    # Sum of each row
```

---

## 🔸 کمینه و بیشینه — `min()`, `max()`

```python
print(x.min())         # Minimum value
print(x.max())         # Maximum value
```

با تعیین بعد (dim) می‌توان مقادیر و ایندکس را نیز دریافت کرد:

```python
values, indices = x.max(dim=1)
print(values)          # Max values per row
print(indices)         # Index of max values per row
```

---

## 🔸 میانه — `median()` و `mode()`

```python
x = torch.tensor([1, 3, 3, 6, 7, 8, 9])

# Median
print(x.median())

# Mode (requires integer tensor)
from torch import mode
print(x.mode())
```

---

## 🔸 ایندکس بزرگ‌ترین/کوچک‌ترین مقدار — `argmax()`, `argmin()`

```python
x = torch.tensor([[10, 20, 15],
                  [5, 8, 2]])

print(x.argmax())       # Index of max value (flattened)
print(x.argmin())       # Index of min value

# Along specific dimension
print(x.argmax(dim=1))  # Max index per row
```

---

## 📌 نکته: خروجی‌ها معمولاً Tensor هستند و برای تبدیل به عدد می‌توان از `.item()` استفاده کرد.

```python
m = x.max()
print(m.item())     # Converts to Python float
```

---

## 📘 تمرین پیشنهادی:

۱. یک Tensor تصادفی با شکل `[4, 5]` بسازید و میانگین هر ستون را حساب کنید.
۲. Tensor یک‌بعدی شامل اعداد تکراری بسازید و `mode` آن را پیدا کنید.
۳. از `argmax` استفاده کنید تا اندیس بزرگ‌ترین عنصر هر ردیف یک Tensor را نمایش دهید.

---

