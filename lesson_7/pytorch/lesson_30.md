
## 🔹 صفحه ۳۰ — مقایسه CNN با مدل خطی و ذخیره‌سازی مدل کانولوشنی

📌 شامل: تفاوت عملکرد CNN با Linear، بررسی تعداد پارامتر، ذخیره/بارگذاری مدل

---

### 🎯 چرا مقایسه بین CNN و Linear مهم است؟

در ابتدای یادگیری، اغلب مدل‌ها را با `nn.Linear` ساده شروع می‌کنیم.
اما برای داده‌های تصویری، استفاده از CNN معمولاً:

* دقت را **افزایش می‌دهد**
* **تعداد پارامترها** را کاهش می‌دهد
* ساختار تصویر را **بهتر می‌فهمد**

---

## 🔸 مدل ساده‌ی خطی برای MNIST

```python
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

✅ این مدل بدون استفاده از `Conv2d` است و صرفاً همه‌ی پیکسل‌ها را با هم متصل می‌کند.

---

## 🔸 مقایسه تعداد پارامترها

```python
cnn = ...   # CNN model from previous page
linear = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

cnn_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
linear_params = sum(p.numel() for p in linear.parameters() if p.requires_grad)

print("CNN Params:", cnn_params)
print("Linear Params:", linear_params)
```

نتیجه نمونه:

```
CNN Params: 119210  
Linear Params: 101770
```

✅ CNN معمولاً پارامتر بیشتری دارد ولی دقت بیشتری نیز دارد (چون ساختار فضایی را درک می‌کند)

---

## 🔸 مقایسه دقت در آموزش

با اجرای هر دو مدل روی داده‌ی MNIST (مثلاً ۵ تا ۱۰ epoch)، خواهید دید:

| مدل    | دقت تقریبی | نقاط قوت              |
| ------ | ---------- | --------------------- |
| Linear | \~91%      | ساده و سریع           |
| CNN    | \~97%      | فهم بهتر ساختار تصویر |

---

## 🔸 ذخیره‌سازی مدل CNN

```python
torch.save(cnn.state_dict(), "cnn_mnist.pth")
```

✅ فقط وزن‌ها ذخیره می‌شوند

---

## 🔸 بارگذاری مدل CNN

```python
# Re-define model with same structure
cnn_loaded = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(8, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16*7*7, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

cnn_loaded.load_state_dict(torch.load("cnn_mnist.pth"))
cnn_loaded.eval()
```

---

## 🔍 پیشنهاد: ترکیب CNN با تکنیک‌های دیگر

| روش          | کاربرد                                |
| ------------ | ------------------------------------- |
| Dropout      | جلوگیری از overfitting                |
| BatchNorm    | تسریع آموزش و پایداری بیشتر           |
| Augmentation | افزایش تنوع داده با transforms تصویری |

(در فصل‌های بعدی به آن‌ها خواهیم پرداخت)

---

## 📘 تمرین پیشنهادی:

۱. یک مدل ساده Linear و یک CNN بسازید
۲. هر دو را روی MNIST آموزش دهید
۳. دقت نهایی، سرعت آموزش و اندازه پارامترها را مقایسه کنید
۴. مدل CNN را ذخیره و سپس بارگذاری کنید و عملکرد آن را تست کنید

---

✅ فصل ششم نیز با موفقیت به پایان رسید!
در فصل هفتم، نگاهی خواهیم داشت به **مباحث تکمیلی و پرکاربرد در PyTorch**
📌 شامل: Dropout, BatchNorm, Flatten دقیق‌تر, Reshape, و تمرین‌های نهایی

---


