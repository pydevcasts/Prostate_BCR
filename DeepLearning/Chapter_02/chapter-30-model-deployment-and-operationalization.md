# فصل ۳۰ 🚀 استقرار مدل (Deployment) و عملیاتی‌سازی

## 🎯 اهداف فصل

تا اینجا یاد گرفتیم که چگونه مدل‌های یادگیری عمیق بسازیم، آموزش دهیم و ارزیابی کنیم. اما ساخت یک مدل با دقت خوب تنها نیمی از کار است. نیمه‌ی دیگر این است که این مدل را از محیط آزمایشگاهی بیرون بیاوریم و در قالب یک سرویس واقعی در دسترس کاربران قرار دهیم. این مرحله را Deployment یا استقرار مدل می‌نامند. در پایان این فصل خواهید توانست مدل PyTorch را به روش‌های مختلف ذخیره و بارگذاری کنید، یک API ساده با FastAPI بسازید که مدل شما را سرویس دهد، مدل را برای استقرار بهینه‌سازی کنید، مدل را با Docker کانتینریزه کنید، و نکات عملی مانیتورینگ و نگهداری مدل در تولید را بشناسید.

---

# 💾 ذخیره و بارگذاری مدل در PyTorch

قبل از هر چیز باید مدل را به‌درستی ذخیره کنیم. PyTorch دو روش اصلی برای ذخیره‌ی مدل دارد. روش اول ذخیره‌ی فقط وزن‌ها (State Dict) است که روش پیشنهادی PyTorch است، چون فایل کوچک‌تری تولید می‌کند و به کد معماری مدل وابستگی کمتری دارد. روش دوم ذخیره‌ی کل مدل است که برای استفاده‌ی سریع مناسب است اما به نسخه‌ی دقیق PyTorch و ساختار کلاس‌ها وابسته است. علاوه بر این، برای استقرار تولیدی می‌توان مدل را به فرمت TorchScript تبدیل کرد که مستقل از کد پایتون اجرا می‌شود.

```python
import torch
import torch.nn as nn
import torchvision.models as models
import os

# ---- روش ۱: ذخیره و بارگذاری State Dict (توصیه‌شده) ----
print("=== روش ۱: State Dict ===\n")

# فرض: مدل آموزش‌دیده داریم
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 10)

# ذخیره
save_path = './model_state_dict.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'model_architecture': 'resnet18',
    'num_classes': 10,
    'input_size': (224, 224),
    'training_info': {
        'best_val_acc': 92.5,
        'epochs_trained': 30
    }
}, save_path)
print(f"مدل ذخیره شد: {save_path}")
print(f"حجم فایل: {os.path.getsize(save_path) / 1024 / 1024:.1f} MB")

# بارگذاری
checkpoint = torch.load(save_path, map_location='cpu')
loaded_model = models.resnet18(pretrained=False)
loaded_model.fc = nn.Linear(512, checkpoint['num_classes'])
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model.eval()
print(f"مدل بارگذاری شد | دقت ذخیره‌شده: {checkpoint['training_info']['best_val_acc']}%")

# ---- روش ۲: TorchScript برای استقرار تولیدی ----
print("\n=== روش ۲: TorchScript ===\n")

# Trace: مدل را با یک ورودی نمونه اجرا می‌کنیم
dummy_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(loaded_model, dummy_input)
torch.jit.save(traced_model, './model_traced.pt')
print("مدل TorchScript ذخیره شد: model_traced.pt")
print("این مدل بدون نیاز به کد پایتون قابل بارگذاری است")

# بارگذاری TorchScript
production_model = torch.jit.load('./model_traced.pt')
production_model.eval()
with torch.no_grad():
    test_output = production_model(dummy_input)
print(f"شکل خروجی: {test_output.shape}")
print("مدل TorchScript با موفقیت اجرا شد")
```

---

# ⚡ بهینه‌سازی مدل برای استقرار

قبل از استقرار، چند تکنیک بهینه‌سازی وجود دارد که می‌توانند سرعت و حافظه‌ی مورد نیاز مدل را بهبود دهند. کوانتیزاسیون (Quantization) وزن‌های مدل را از دقت ۳۲ بیت به ۸ بیت تبدیل می‌کند که حجم مدل را به یک‌چهارم کاهش می‌دهد و سرعت را تا ۲ تا ۴ برابر افزایش می‌دهد. هرس (Pruning) اتصالات کم‌اهمیت را حذف می‌کند.

```python
import torch
import torch.nn as nn
import torch.quantization
import time

# ---- کوانتیزاسیون پس از آموزش ----
print("=== بهینه‌سازی: Quantization ===\n")

model_fp32 = models.resnet18(pretrained=False)
model_fp32.fc = nn.Linear(512, 10)
model_fp32.eval()

# اندازه‌گیری سرعت مدل اصلی
dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    start = time.time()
    for _ in range(100):
        _ = model_fp32(dummy)
    fp32_time = (time.time() - start) / 100 * 1000

fp32_size = sum(p.numel() * 4 for p in model_fp32.parameters()) / 1024 / 1024

# کوانتیزاسیون دینامیک (ساده‌ترین روش)
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {nn.Linear},
    dtype=torch.qint8
)

with torch.no_grad():
    start = time.time()
    for _ in range(100):
        _ = model_int8(dummy)
    int8_time = (time.time() - start) / 100 * 1000

# ذخیره و بررسی اندازه
torch.jit.save(torch.jit.script(model_int8), './model_int8.pt')
int8_size = os.path.getsize('./model_int8.pt') / 1024 / 1024

print(f"مدل FP32 — حجم: {fp32_size:.1f} MB | سرعت: {fp32_time:.2f} ms")
print(f"مدل INT8  — حجم: {int8_size:.1f} MB | سرعت: {int8_time:.2f} ms")
print(f"کاهش حجم: {100*(1-int8_size/fp32_size):.1f}%")
print(f"افزایش سرعت: {fp32_time/int8_time:.1f}x")
```

---

# 🌐 ساخت API با FastAPI

FastAPI یکی از محبوب‌ترین فریم‌ورک‌های پایتون برای ساخت API است که با سرعت بالا و پشتیبانی خودکار از مستندات، انتخاب ایده‌آلی برای سرویس‌دهی مدل‌های یادگیری عمیق است.

```python
# ---- کد کامل API با FastAPI ----
# ذخیره این کد در فایل main.py و اجرا با: uvicorn main:app --reload

api_code = '''
# نصب: pip install fastapi uvicorn python-multipart pillow

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
import time
import logging
from typing import List, Dict

# تنظیم لاگ‌گذاری
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- تعریف اپلیکیشن FastAPI ----
app = FastAPI(
    title="سرویس طبقه‌بندی تصویر",
    description="API برای طبقه‌بندی تصاویر با مدل یادگیری عمیق",
    version="1.0.0"
)

# فعال‌سازی CORS برای دسترسی از مرورگر
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---- بارگذاری مدل در هنگام راه‌اندازی ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
class_names = []

@app.on_event("startup")
async def load_model():
    global model, class_names
    logger.info("در حال بارگذاری مدل...")

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 10)

    checkpoint = torch.load("model_state_dict.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    class_names = [f"کلاس_{i}" for i in range(10)]
    logger.info(f"مدل بارگذاری شد روی: {device}")

# تبدیل‌های پیش‌پردازش
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---- مدل‌های پاسخ ----
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_5: List[Dict[str, float]]
    inference_time_ms: float

class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool

# ---- اندپوینت‌ها ----
@app.get("/", summary="صفحه‌ی خوش‌آمد")
async def root():
    return {"message": "به سرویس طبقه‌بندی تصویر خوش آمدید", "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse, summary="بررسی وضعیت سرویس")
async def health_check():
    return HealthResponse(
        status="healthy",
        device=str(device),
        model_loaded=model is not None
    )

@app.post("/predict", response_model=PredictionResponse, summary="پیش‌بینی کلاس تصویر")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="فایل باید تصویر باشد")

    if model is None:
        raise HTTPException(status_code=503, detail="مدل هنوز بارگذاری نشده")

    try:
        # خواندن و پیش‌پردازش تصویر
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        # پیش‌بینی
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        inference_time = (time.time() - start_time) * 1000

        # پنج کلاس برتر
        top5_probs, top5_indices = torch.topk(probabilities, 5)
        top5 = [
            {class_names[idx.item()]: round(prob.item(), 4)}
            for idx, prob in zip(top5_indices, top5_probs)
        ]

        predicted_idx = top5_indices[0].item()

        logger.info(f"پیش‌بینی: {class_names[predicted_idx]} "
                   f"({top5_probs[0].item():.2%}) در {inference_time:.1f}ms")

        return PredictionResponse(
            predicted_class=class_names[predicted_idx],
            confidence=round(top5_probs[0].item(), 4),
            top_5=top5,
            inference_time_ms=round(inference_time, 2)
        )

    except Exception as e:
        logger.error(f"خطا در پیش‌بینی: {e}")
        raise HTTPException(status_code=500, detail=f"خطای داخلی: {str(e)}")

@app.post("/predict/batch", summary="پیش‌بینی دسته‌ای")
async def predict_batch(files: List[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="حداکثر ۱۰ تصویر در هر درخواست")

    results = []
    for file in files:
        try:
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            input_tensor = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]

            predicted_idx = probs.argmax().item()
            results.append({
                "filename": file.filename,
                "predicted_class": class_names[predicted_idx],
                "confidence": round(probs[predicted_idx].item(), 4)
            })
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return {"predictions": results, "total": len(results)}
'''

# ذخیره کد API در فایل
with open('./main.py', 'w', encoding='utf-8') as f:
    f.write(api_code)
print("کد API در فایل main.py ذخیره شد")
print("\nبرای اجرای API:")
print("  uvicorn main:app --reload --host 0.0.0.0 --port 8000")
print("\nمستندات خودکار API در:")
print("  http://localhost:8000/docs")
```

---

# 🐳 کانتینریزه کردن با Docker

Docker به شما اجازه می‌دهد مدل و تمام وابستگی‌هایش را در یک محیط مستقل بسته‌بندی کنید که روی هر سروری قابل اجراست.

```python
# ---- ساخت فایل‌های Docker ----

dockerfile_content = """# استفاده از Python 3.10 Slim به‌عنوان پایه
FROM python:3.10-slim

# تنظیم متغیر محیطی برای جلوگیری از بافر شدن لاگ
ENV PYTHONUNBUFFERED=1

# تنظیم دایرکتوری کاری
WORKDIR /app

# کپی و نصب وابستگی‌ها (این لایه کش می‌شود اگر requirements.txt تغییر نکند)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# کپی کد برنامه و مدل
COPY main.py .
COPY model_state_dict.pt .

# معرفی پورت
EXPOSE 8000

# دستور اجرا
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

requirements_content = """fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
torch==2.1.0
torchvision==0.16.0
pillow==10.1.0
pydantic==2.4.2
"""

docker_compose_content = """version: '3.8'

services:
  model-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs
    environment:
      - LOG_LEVEL=info
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
"""

with open('./Dockerfile', 'w') as f:
    f.write(dockerfile_content)
with open('./requirements.txt', 'w') as f:
    f.write(requirements_content)
with open('./docker-compose.yml', 'w') as f:
    f.write(docker_compose_content)

print("فایل‌های Docker ساخته شدند:\n")
print("  📄 Dockerfile")
print("  📄 requirements.txt")
print("  📄 docker-compose.yml")
print("\nدستورات Docker:")
print("  docker build -t model-api .")
print("  docker run -p 8000:8000 model-api")
print("  docker-compose up -d   ← اجرای پس‌زمینه")
```

---

# 📊 مانیتورینگ مدل در تولید

استقرار مدل پایان کار نیست؛ باید مطمئن شوید که مدل در طول زمان به‌درستی کار می‌کند. مهم‌ترین چیزهایی که باید مانیتور کنید عبارتند از: زمان پاسخ‌دهی (Latency)، نرخ خطا، توزیع ورودی‌ها (که اگر تغییر کند ممکن است مدل دیگر مناسب نباشد)، و توزیع پیش‌بینی‌ها. اگر مدل در طول زمان روی داده‌های واقعی عملکرد ضعیف‌تری داشت، این پدیده را Data Drift یا Concept Drift می‌نامند و نشانه‌ای است که مدل نیاز به بازآموزش دارد.

```python
# ---- یک سیستم ساده‌ی مانیتورینگ ----
import time
import json
from collections import deque, defaultdict
from datetime import datetime

class ModelMonitor:
    """سیستم ساده برای مانیتورینگ مدل در تولید"""

    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        self.class_counts = defaultdict(int)
        self.total_requests = 0
        self.start_time = datetime.now()

    def log_prediction(self, predicted_class: str, confidence: float,
                        latency_ms: float, is_error: bool = False):
        self.total_requests += 1
        self.latencies.append(latency_ms)
        self.errors.append(1 if is_error else 0)

        if not is_error:
            self.predictions.append(confidence)
            self.class_counts[predicted_class] += 1

    def get_metrics(self) -> dict:
        if not self.latencies:
            return {}

        latencies_list = list(self.latencies)
        confidences_list = list(self.predictions)
        errors_list = list(self.errors)

        return {
            "uptime_hours": (datetime.now() - self.start_time).seconds / 3600,
            "total_requests": self.total_requests,
            "error_rate": sum(errors_list) / len(errors_list) if errors_list else 0,
            "latency": {
                "mean_ms": sum(latencies_list) / len(latencies_list),
                "max_ms": max(latencies_list),
                "p95_ms": sorted(latencies_list)[int(0.95 * len(latencies_list))]
            },
            "confidence": {
                "mean": sum(confidences_list) / len(confidences_list) if confidences_list else 0,
                "low_confidence_rate": sum(1 for c in confidences_list if c < 0.5) / len(confidences_list) if confidences_list else 0
            },
            "class_distribution": dict(self.class_counts)
        }

    def check_alerts(self) -> list:
        alerts = []
        metrics = self.get_metrics()

        if not metrics:
            return alerts

        if metrics["error_rate"] > 0.05:
            alerts.append(f"⚠️ نرخ خطا بالاست: {metrics['error_rate']:.1%}")

        if metrics["latency"]["p95_ms"] > 500:
            alerts.append(f"⚠️ تأخیر P95 بالاست: {metrics['latency']['p95_ms']:.0f}ms")

        if metrics["confidence"]["low_confidence_rate"] > 0.3:
            alerts.append(f"⚠️ نرخ اطمینان پایین بالاست: {metrics['confidence']['low_confidence_rate']:.1%}")

        return alerts


# آزمایش سیستم مانیتورینگ
monitor = ModelMonitor()
import random

print("=== شبیه‌سازی مانیتورینگ مدل در تولید ===\n")

for i in range(200):
    latency = random.gauss(120, 30)
    confidence = random.gauss(0.75, 0.15)
    confidence = max(0.1, min(0.99, confidence))
    is_error = random.random() < 0.02
    predicted_class = f"کلاس_{random.randint(0, 9)}"
    monitor.log_prediction(predicted_class, confidence, latency, is_error)

metrics = monitor.get_metrics()
print(f"کل درخواست‌ها: {metrics['total_requests']}")
print(f"نرخ خطا: {metrics['error_rate']:.1%}")
print(f"تأخیر میانگین: {metrics['latency']['mean_ms']:.1f} ms")
print(f"تأخیر P95: {metrics['latency']['p95_ms']:.1f} ms")
print(f"اطمینان میانگین: {metrics['confidence']['mean']:.2f}")
print(f"نرخ اطمینان پایین: {metrics['confidence']['low_confidence_rate']:.1%}")

alerts = monitor.check_alerts()
if alerts:
    print("\nهشدارها:")
    for alert in alerts:
        print(f"  {alert}")
else:
    print("\nسیستم در وضعیت عادی است. ✅")
```

---

# ❓ سؤالات تستی

### سؤال ۱

چرا برای استقرار تولیدی، ذخیره‌ی State Dict توصیه‌شده‌تر از ذخیره‌ی کل مدل است؟

الف) چون State Dict حجم کمتری دارد

ب) چون State Dict فقط وزن‌ها را ذخیره می‌کند و به نسخه‌ی خاص PyTorch یا ساختار کلاس وابسته نیست؛ این باعث می‌شود پایدارتر و قابل‌حمل‌تر باشد

ج) چون ذخیره‌ی کل مدل در PyTorch کار نمی‌کند

د) چون State Dict سریع‌تر بارگذاری می‌شود

### سؤال ۲

کوانتیزاسیون مدل چه تأثیری بر کیفیت پیش‌بینی دارد؟

الف) کیفیت پیش‌بینی را کاملاً از بین می‌برد

ب) معمولاً کاهش بسیار جزئی در دقت (اغلب کمتر از یک درصد) در برابر کاهش قابل‌توجه حجم (حدود ۷۵ درصد) و افزایش سرعت (۲ تا ۴ برابر) ایجاد می‌کند

ج) کیفیت را بهبود می‌بخشد

د) هیچ تأثیری بر کیفیت یا سرعت ندارد

---

## ✅ پاسخ‌ها

**پاسخ سؤال ۱: گزینه ب**

وقتی کل مدل ذخیره می‌شود، PyTorch کل شیء مدل را با Pickle سریال می‌کند که شامل ارجاع به کلاس‌های پایتون می‌شود. اگر ساختار کلاس یا نسخه‌ی PyTorch تغییر کند، بارگذاری ممکن است با خطا مواجه شود. ذخیره‌ی State Dict این وابستگی را حذف می‌کند چون فقط یک دیکشنری از نام لایه به تنسور وزن است.

**پاسخ سؤال ۲: گزینه ب**

کوانتیزاسیون به ۸ بیت در اکثر مدل‌های بزرگ تأثیر بسیار کمی بر دقت دارد (معمولاً کمتر از ۰.۵ درصد کاهش). این معامله‌ی کاهش ۷۵ درصدی حجم و افزایش ۲ تا ۴ برابری سرعت در برابر کاهش جزئی دقت، برای اکثر کاربردهای تولیدی کاملاً ارزنده است.

---

# 📝 خلاصه فصل

در این فصل چرخه‌ی کامل استقرار یک مدل یادگیری عمیق را بررسی کردیم. روش‌های مختلف ذخیره‌ی مدل (State Dict، کل مدل و TorchScript) را مقایسه کردیم. تکنیک کوانتیزاسیون را برای کاهش حجم و افزایش سرعت آموختیم. یک API کامل با FastAPI ساختیم که تصویر دریافت می‌کند و کلاس آن را برمی‌گرداند. مدل را با Docker کانتینریزه کردیم تا روی هر سروری قابل اجرا باشد. و در نهایت یک سیستم مانیتورینگ ساده برای پایش سلامت مدل در تولید پیاده‌سازی کردیم.

---

# 🎯 تمرین‌های پایان فصل

۱. مدل CIFAR-10 از فصل‌های قبل را کوانتیزه کنید و کاهش دقت آن را اندازه بگیرید.
۲. یک اندپوینت جدید به API اضافه کنید که URL تصویر را دریافت کند و دانلود و پیش‌بینی را انجام دهد.
۳. Dockerfile را بهینه کنید تا از Multi-stage Build استفاده کند.
۴. یک داشبورد ساده با Streamlit بسازید که کاربر بتواند تصویر آپلود کند و نتیجه را ببیند.
۵. Load Testing ساده‌ای با ابزار locust انجام دهید تا ببینید API شما در چه تعداد درخواست همزمان گلوگاه ایجاد می‌کند.

---

در فصل ۳۱ به **جمع‌بندی کتاب، مسیر یادگیری بعدی و راهنمای ورود به بازار کار** می‌پردازیم و نقشه‌ی راه کاملی برای تبدیل‌شدن به یک مهندس یادگیری عمیق حرفه‌ای ارائه می‌دهیم.
