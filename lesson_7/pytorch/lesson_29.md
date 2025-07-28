
## 🔹 صفحه ۲۹ — ساخت یک شبکه‌ی CNN ساده برای طبقه‌بندی MNIST

📌 شامل: `Conv2d`, `ReLU`, `MaxPool2d`, `Flatten`, و `Linear`

---

### 🎯 هدف ما چیست؟

ساخت یک شبکه‌ی کانولوشنی ساده برای تشخیص ارقام دست‌نویس (MNIST)
با ترکیب چند لایه‌ی کانولوشن + Pooling + لایه‌های Fully Connected

---

## 🔸 معماری پیشنهادی

```
Input → Conv(1→8) → ReLU → MaxPool  
      → Conv(8→16) → ReLU → MaxPool  
      → Flatten → Linear(16×7×7 → 128) → ReLU  
      → Linear(128 → 10 classes)
```

---

## 🔸 پیاده‌سازی مدل با `nn.Sequential`

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(8, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),

    nn.Linear(16 * 7 * 7, 128),
    nn.ReLU(),

    nn.Linear(128, 10)  # 10 output classes
)
```

✅ ورودی MNIST: `[batch, 1, 28, 28]`
✅ بعد از دو مرحله Pooling → ابعاد می‌شود `[batch, 16, 7, 7]`

---

## 🔸 بررسی ابعاد گام به گام

```python
x = torch.rand(1, 1, 28, 28)
out = model[0](x)        # Conv(1→8)
print(out.shape)         # [1, 8, 28, 28]

out = model[2](out)      # MaxPool
print(out.shape)         # [1, 8, 14, 14]

out = model[3](out)      # Conv(8→16)
out = model[5](out)      # MaxPool
print(out.shape)         # [1, 16, 7, 7]

out = model[6](out)      # Flatten
print(out.shape)         # [1, 784]

out = model[7](out)      # Linear
print(out.shape)         # [1, 128]
```

---

## 🔸 استفاده در آموزش

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
```

سپس در حلقه آموزش:

```python
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)

    loss = loss_fn(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 🔍 نکات کلیدی این معماری

| بخش         | توضیح                                     |
| ----------- | ----------------------------------------- |
| `Conv2d`    | استخراج ویژگی از تصویر                    |
| `ReLU`      | غیرفخطی‌سازی                              |
| `MaxPool2d` | کاهش ابعاد برای سرعت و تمرکز روی ویژگی‌ها |
| `Flatten`   | تبدیل داده 3بعدی به 1بعدی                 |
| `Linear`    | تبدیل ویژگی‌ها به خروجی طبقه‌بندی         |

---

## 📘 تمرین پیشنهادی:

۱. مدل بالا را اجرا کنید و آموزش دهید (۵ تا ۱۰ epoch روی MNIST)
۲. مقدار `out_channels` را تغییر دهید و دقت نهایی را بررسی کنید
۳. `Flatten` را با `.view()` جایگزین کنید و نتیجه را مقایسه کنید

---

✅ در صفحه بعدی (۳۰)، فصل ششم را جمع‌بندی می‌کنیم با:
**مقایسه‌ی مدل CNN با مدل ساده `Linear`, بررسی تعداد پارامترها، و نحوه ذخیره‌سازی مدل کانولوشنی**

---

