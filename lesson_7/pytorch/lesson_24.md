
# 🟢 فصل چهارم: استفاده از GPU در PyTorch

## 🔹 صفحه ۲۴ — انتقال داده و مدل به CUDA (GPU) برای سرعت بیشتر

📌 شامل: استفاده از `.to("cuda")`, بررسی دسترسی به GPU, مقایسه سرعت CPU و CUDA

---

### 🎯 چرا از GPU استفاده کنیم؟

**مدل‌های یادگیری ماشین و عمیق**، به‌ویژه در مسائل تصویری و شبکه‌های بزرگ، روی **CPU کند اجرا می‌شوند**.
✅ استفاده از **GPU (CUDA)** باعث می‌شود:

* آموزش مدل‌ها بسیار سریع‌تر انجام شود
* عملیات ماتریسی سنگین با سرعت بالا پردازش شوند

---

## 🔸 بررسی در دسترس بودن CUDA

```python
import torch

print(torch.cuda.is_available())  # True if CUDA is available
print(torch.cuda.device_count())  # Number of GPUs
```

خروجی:

```
True  
1
```

✅ اگر `False` بود، یعنی سیستم شما یا گوگل کولب شما GPU فعال ندارد.

---

## 🔸 تعیین دستگاه (device)

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```

---

## 🔸 انتقال مدل به CUDA

```python
model = MyModel()             # Define your model
model.to(device)              # Move model to GPU if available
```

---

## 🔸 انتقال داده به CUDA

```python
x = torch.rand(32, 10)
y = torch.rand(32, 1)

x = x.to(device)
y = y.to(device)
```

---

## 🔸 اجرای کامل روی GPU

```python
output = model(x)
loss = loss_fn(output, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

✅ حتماً **هم مدل و هم داده‌ها** باید روی یک device باشند (یا هر دو روی CPU یا هر دو روی CUDA)

---

## 🔸 انتقال برعکس به CPU

```python
output_cpu = output.to("cpu")  # For visualization or saving
```

---

## 🔍 نکات مهم در استفاده از CUDA

| نکته                                  | توضیح                                                 |
| ------------------------------------- | ----------------------------------------------------- |
| بررسی `is_available()`                | همیشه قبل از استفاده بررسی شود                        |
| مدل و داده باید هم‌زمان روی GPU باشند | تا خطا ندهد                                           |
| انتقال خروجی به CPU                   | برای مشاهده یا ذخیره لازم است                         |
| در کولب                               | از منوی Runtime → Change Runtime Type → GPU فعال کنید |

---

## 📘 تمرین پیشنهادی:

۱. بررسی کنید آیا سیستم شما CUDA دارد یا نه
۲. یک مدل را به CUDA منتقل کرده و آموزش دهید
۳. تفاوت سرعت اجرای چند epoch روی CPU و GPU را با `time.time()` مقایسه کنید
۴. خروجی را از CUDA به CPU برگردانید و با `.detach().numpy()` به NumPy تبدیل کنید

---

✅ در صفحه بعدی به سراغ فصل پنجم و **عملیات پیشرفته روی Tensorها** خواهیم رفت
📌 شامل: `stack`, `cat`, `split`, `chunk`, و کاربرد عملی هرکدام

---

