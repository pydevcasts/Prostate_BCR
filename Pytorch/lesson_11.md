
## 🔹 صفحه ۱۱ — تبدیل بین PyTorch و NumPy

📌 شامل: `.numpy()` و `torch.from_numpy()`

---

### 🎯 چرا تبدیل بین PyTorch و NumPy مهم است؟

خیلی وقت‌ها در یادگیری ماشین و علم داده، از کتابخانه‌ی **NumPy** برای پردازش اولیه یا تحلیل داده استفاده می‌کنیم، و از **PyTorch** برای ساخت مدل.

برای هماهنگی این دو ابزار، PyTorch امکان تبدیل مستقیم **بین Tensor و آرایه NumPy** را فراهم کرده است، بدون نیاز به کپی‌برداری دستی.

---

## 🔸 تبدیل Tensor به NumPy Array — `.numpy()`

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])

# Convert to NumPy array
np_array = x.numpy()

print(np_array)
```

خروجی:

```
[1. 2. 3.]
```

✅ این تبدیل **مستقیم** است و اگر `x` تغییر کند، `np_array` نیز تغییر خواهد کرد (و بالعکس)، چون حافظه مشترک دارند.

---

## 🔸 نکته: `.numpy()` فقط روی Tensorهای CPU قابل استفاده است!

```python
x = torch.tensor([1.0]).cuda()

# This will raise an error:
# x.numpy()
```

✅ باید ابتدا آن را به CPU منتقل کنید:

```python
x = x.to("cpu")
x.numpy()
```

---

## 🔸 تبدیل NumPy Array به Tensor — `torch.from_numpy()`

```python
import numpy as np

np_array = np.array([10, 20, 30])

# Convert to PyTorch tensor
x = torch.from_numpy(np_array)

print(x)
```

خروجی:

```
tensor([10, 20, 30])
```

✅ همانند قبل، **حافظه مشترک** است. اگر یکی تغییر کند، آن یکی هم تغییر خواهد کرد.

---

## 🔸 تبدیل کامل و امن (بدون حافظه مشترک)

اگر بخواهید از حافظه جدا استفاده کنید (کپی کامل)، می‌توانید `.clone()` یا `.detach()` استفاده کنید:

```python
# Safe copy to NumPy
x = torch.tensor([1.0, 2.0])
x_np = x.clone().detach().numpy()
```

---

## 🔍 جمع‌بندی سریع:

| عملیات         | تابع                 | نکته                         |
| -------------- | -------------------- | ---------------------------- |
| Tensor → NumPy | `.numpy()`           | فقط روی CPU قابل اجراست      |
| NumPy → Tensor | `torch.from_numpy()` | نیاز به نوع داده سازگار دارد |
| جدا کردن حافظه | `.clone().detach()`  | برای کپی مستقل               |

---

## 📘 تمرین پیشنهادی:

۱. یک NumPy آرایه بسازید و آن را به Tensor تبدیل کنید و نوع داده‌اش را بررسی کنید.
۲. یک Tensor با `.numpy()` به آرایه NumPy تبدیل کنید، سپس یکی از عناصر آن را تغییر دهید و ببینید آیا در دیگری هم تغییر می‌کند؟
۳. از `.clone()` استفاده کنید تا یک نسخه مستقل بسازید و مطمئن شوید ارتباط بین آن‌ها قطع شده است.

---
