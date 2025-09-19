## 🔹 صفحه ۱۹ — طراحی حلقه آموزش (Training Loop) ساده در PyTorch

📌 شامل: اجرای چند epoch، محاسبه loss در هر مرحله، به‌روزرسانی پارامترها

---

### 🎯 چرا حلقه آموزش اهمیت دارد؟

برای اینکه مدل یاد بگیرد، باید آموزش آن را **در چند مرحله تکراری** انجام دهیم.
هر مرحله از آموزش به‌صورت زیر انجام می‌شود:

1. پیش‌بینی با مدل (`forward pass`)
2. محاسبه خطا (`loss`)
3. محاسبه گرادیان‌ها (`backward`)
4. بروزرسانی وزن‌ها (`optimizer.step()`)
5. تکرار مراحل بالا به تعداد مشخصی از **epoch**ها

---

## 🔸 ساخت یک مدل ساده

```python
import torch
import torch.nn as nn

# Model with 10 inputs and 1 output
model = nn.Linear(10, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

---

## 🔸 داده‌های فرضی

```python
# Simulated dataset
x = torch.rand(100, 10)  # 100 samples, 10 features
y = torch.rand(100, 1)   # 100 targets
```

---

## 🔸 اجرای Training Loop

```python
num_epochs = 20

for epoch in range(num_epochs):
    # Forward
    y_pred = model(x)
    
    # Compute loss
    loss = loss_fn(y_pred, y)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print loss every few epochs
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
```

خروجی نمونه:

```
Epoch 5/20, Loss: 0.1321  
Epoch 10/20, Loss: 0.0854  
Epoch 15/20, Loss: 0.0433  
Epoch 20/20, Loss: 0.0129
```

✅ مشاهده می‌کنید که مقدار Loss با تکرار epochs کاهش می‌یابد.

---

## 🔍 چه نکاتی در حلقه آموزش مهم هستند؟

| بخش                     | اهمیت                          |
| ----------------------- | ------------------------------ |
| `optimizer.zero_grad()` | پاک کردن گرادیان‌های مرحله قبل |
| `loss.backward()`       | محاسبه گرادیان بر اساس خطا     |
| `optimizer.step()`      | به‌روزرسانی وزن‌های مدل        |
| `epoch`                 | تکرار آموزش برای یادگیری بهتر  |

---

## 🔸 چگونه عملکرد مدل را بسنجیم؟

در پروژه‌های واقعی، بعد از هر epoch معمولاً:

* از داده‌های اعتبارسنجی (validation) برای بررسی یادگیری استفاده می‌شود
* نمودار تغییر Loss در طول زمان رسم می‌شود
  (در صفحات آینده به این‌ها خواهیم پرداخت)

---

## 📘 تمرین پیشنهادی:

۱. حلقه آموزشی بالا را برای `20`، `50` و `100` epoch اجرا کنید و مقدار Loss نهایی را مقایسه کنید.
۲. از `optimizer.Adam` به جای `SGD` استفاده کنید و بررسی کنید آیا سرعت کاهش Loss بیشتر می‌شود یا نه.
۳. مقدار `learning rate` را کم و زیاد کرده و رفتار Loss را بررسی کنید.

---

✅ در صفحه بعدی وارد موضوع مهمی می‌شویم:
**ذخیره و بارگذاری مدل‌ها با `torch.save` و `load_state_dict()`**
📌 کاربردی برای نگهداری مدل و ادامه آموزش در آینده

---

