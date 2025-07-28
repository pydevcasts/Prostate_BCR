## 🔹 صفحه ۲۰ — ذخیره‌سازی و بارگذاری مدل در PyTorch

📌 شامل: `torch.save()`, `torch.load()`, `model.state_dict()`, `load_state_dict()`

---

### 🎯 چرا باید مدل را ذخیره کنیم؟

وقتی مدل آموزش دیده، باید بتوانیم:

* مدل را ذخیره کنیم تا در آینده از آن استفاده کنیم (بدون نیاز به آموزش مجدد)
* آموزش را در زمان دیگری ادامه دهیم
* یا مدل را برای تست یا استقرار (deployment) منتقل کنیم

---

## 🔸 ذخیره مدل با `torch.save()`

در PyTorch می‌توان فقط **پارامترهای آموزش‌پذیر مدل** را ذخیره کرد. برای این کار از `model.state_dict()` استفاده می‌کنیم.

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)

# Save model parameters
torch.save(model.state_dict(), "mymodel.pth")
```

✅ فایل `mymodel.pth` روی دیسک ذخیره می‌شود و فقط شامل وزن‌هاست، نه ساختار کامل مدل.

---

## 🔸 بارگذاری مدل با `load_state_dict()`

ابتدا باید مدل را با همان ساختار تعریف کنیم، سپس وزن‌ها را بارگذاری کنیم.

```python
# Define same model architecture
model2 = nn.Linear(10, 1)

# Load weights into model
model2.load_state_dict(torch.load("mymodel.pth"))

# Now model2 is ready to use with trained weights
model2.eval()  # Optional: switch to evaluation mode
```

---

### 🔍 نکته مهم:

اگر مدل دارای لایه‌ها یا پارامترهایی باشد که در ساختار جدید نیستند، بارگذاری موفق نخواهد بود. ساختار مدل باید کاملاً **همانند نسخه ذخیره‌شده** باشد.

---

## 🔸 ذخیره‌سازی کامل مدل (کمتر توصیه‌شده)

```python
torch.save(model, "full_model.pth")
```

و سپس:

```python
model_loaded = torch.load("full_model.pth")
```

✅ این روش مدل کامل (ساختار + وزن) را ذخیره می‌کند ولی **وابسته به کلاس و نسخه پایتورچ** است. توصیه می‌شود فقط از `state_dict` استفاده کنید.

---

## 🔸 نمایش محتوای `state_dict()`

```python
for name, param in model.state_dict().items():
    print(name, param.shape)
```

نمونه خروجی:

```
weight torch.Size([1, 10])
bias torch.Size([1])
```

---

## 📘 تمرین پیشنهادی:

۱. یک مدل آموزش دیده ذخیره کنید و سپس در برنامه‌ای جداگانه آن را بارگذاری و تست کنید.
۲. از `state_dict` پرینت بگیرید و ساختار لایه‌ها را بررسی کنید.
۳. یک مدل با ساختار متفاوت تعریف کرده و سعی کنید وزن‌ها را بارگذاری کنید؛ خطا را تحلیل کنید.

---

✅ فصل دوم نیز رو به پایان است. در فصل سوم وارد **کار با داده‌ها در PyTorch** خواهیم شد:
📌 استفاده از `Dataset`, `DataLoader`, تقسیم داده‌ها، batch، و آماده‌سازی داده برای مدل

---

