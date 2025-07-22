## آموزش جامع PyTorch (پای‌تورچ)

---

### 1. نصب PyTorch

برای نصب PyTorch به همراه torchvision و torchaudio:

```bash
pip install torch torchvision torchaudio
```

---

### 2. تنسورها در PyTorch

#### 2.1. تعریف تنسورها

```python
import torch

# 1D Tensor
t1 = torch.tensor([1, 2, 3])
# 2D Tensor
t2 = torch.tensor([[1, 2], [3, 4]])
# 3D Tensor
t3 = torch.tensor([[[1], [2]], [[3], [4]]])
```

#### 2.2. ویژگی‌های تنسورها

- **نوع داده:**

```python
t_float = torch.tensor([1, 2], dtype=torch.float32)
```

- **ابعاد و اندازه:**

```python
print(t2.shape)  # torch.Size([2, 2])
print(t2.size(0))  # دسترسی به بعد اول
```

- **تغییر ابعاد:**

```python
t = torch.tensor([1, 2, 3, 4, 5, 6])
t_view = t.view(3, 2)
```

- **تبدیل به GPU:**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_gpu = t.to(device)
```

---

### 3. ایجاد داده‌های تصادفی

#### 3.1. `torch.rand` - توزیع یکنواخت

```python
torch.rand(2, 3)
```

#### 3.2. `torch.randn` - توزیع نرمال

```python
torch.randn(2, 3)
```

#### 3.3. انتقال به GPU

```python
rand_tensor = torch.rand(3, 2).to(device)
```

---

### 4. عملیات ریاضی روی تنسورها

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

sum_ab = a + b
product_ab = a * b
```

---

### 5. ساخت مدل‌های یادگیری عمیق

#### 5.1. تعریف مدل

```python
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc1(x)
```

#### 5.2. تابع هزینه و بهینه‌ساز

```python
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

#### 5.3. آموزش مدل

```python
inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

---

### 6. ذخیره و بارگذاری مدل

```python
torch.save(model.state_dict(), 'model.pth')

model_loaded = SimpleNN()
model_loaded.load_state_dict(torch.load('model.pth'))
```

---

### 7. Transforms و پردازش تصویر

#### 7.1. نصب و وارد کردن

```bash
pip install torchvision
```

```python
import torchvision.transforms as transforms
```

#### 7.2. استفاده از توابع تبدیل

```python
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

#### 7.3. استفاده در DataLoader

```python
from torchvision import datasets
from torch.utils.data import DataLoader

dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

### 8. Autograd و محاسبه گرادیان

#### 8.1. فعال‌سازی گرادیان

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3*x + 1
y.backward()
print(x.grad)
```

#### 8.2. نگهداری گراف

```python
y.backward(retain_graph=True)
```

---

### 9. Dataset و DataLoader سفارشی

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

---

### 10. مدیریت حالت آموزش و تست

```python
model.train()  # فعال‌سازی Dropout و BatchNorm
model.eval()   # غیرفعال‌سازی در زمان تست
```

---

### 11. غیرفعال‌سازی گرادیان در حالت تست

```python
with torch.no_grad():
    predictions = model(inputs)
```

---

### 12. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### 13. Learning Rate Scheduler

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
for epoch in range(20):
    train()
    scheduler.step()
```

---

### 14. استفاده از مدل‌های از پیش آموزش‌دیده

```python
from torchvision import models
resnet = models.resnet18(pretrained=True)
```

---

### 15. ذخیره وضعیت کامل آموزش

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')
```

---

### 16. مقایسه `torch.rand` و `torch.randn`

| تابع    | توزیع   | دامنه     | کاربرد                       |
| ------- | ------- | --------- | ---------------------------- |
| `rand`  | یکنواخت | [0, 1)    | مقداردهی اولیه ساده          |
| `randn` | نرمال   | میانگین 0 | یادگیری آماری، وزن‌های نرمال |

---


