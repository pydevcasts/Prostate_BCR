# فصل ۲۴ 🔧 Fine-tuning پیشرفته‌ی مدل‌های زبانی بزرگ: LoRA و QLoRA

## 🎯 اهداف فصل

در فصل قبل Fine-tuning استاندارد BERT را یاد گرفتیم که تمام پارامترهای مدل در آموزش شرکت می‌کنند. اما برای مدل‌های زبانی بزرگ‌تر مثل LLaMA-2 با هفت میلیارد پارامتر یا GPT-3 با ۱۷۵ میلیارد پارامتر، این رویکرد به حافظه‌ی گرافیکی (VRAM) بسیار زیادی نیاز دارد که برای اکثر توسعه‌دهندگان در دسترس نیست. در این فصل تکنیک‌های PEFT (Parameter-Efficient Fine-Tuning) را یاد می‌گیریم که آموزش مدل‌های بسیار بزرگ را حتی با منابع محدود ممکن می‌سازند. در پایان این فصل خواهید توانست مشکل اصلی Fine-tuning کامل مدل‌های بزرگ را توضیح دهید، مفهوم LoRA را با استدلال ریاضی درک کنید، تفاوت LoRA و QLoRA را بشناسید، مفهوم Instruction Tuning را توضیح دهید، و یک مدل کوچک را با LoRA روی دیتاست اختصاصی Fine-tune کنید.

---

# 💸 چالش Fine-tuning کامل مدل‌های بزرگ

وقتی بخواهیم یک مدل با هفت میلیارد پارامتر را Fine-tune کنیم، تنها نگه‌داشتن وزن‌های آن در حافظه به ۱۴ گیگابایت (با دقت FP16) نیاز دارد. اما در طول آموزش، علاوه بر وزن‌های مدل، باید گرادیان‌ها (۱۴ گیگابایت دیگر) و وضعیت بهینه‌ساز Adam (۲۸ گیگابایت دیگر برای ذخیره‌ی دو لحظه‌ی گرادیان) هم در حافظه نگه‌داشته شوند. این یعنی فقط برای آموزش یک مدل هفت‌میلیارد پارامتری به حدود ۵۶ گیگابایت VRAM نیاز است که حتی بهترین کارت‌های گرافیک مصرف‌کننده هم آن را ندارند.

این واقعیت باعث می‌شود که تنها شرکت‌های بزرگ با زیرساخت GPU گران‌قیمت بتوانند LLM بسازند یا Fine-tune کنند. اما تکنیک‌های PEFT این معادله را تغییر داده‌اند و آموزش مدل‌های بزرگ را با منابع بسیار محدودتر ممکن کرده‌اند.

---

# 📐 LoRA: تقریب ماتریس وزن با رتبه‌ی پایین

LoRA که مخفف Low-Rank Adaptation است، در سال ۲۰۲۱ توسط هو و همکارانش معرفی شد. ایده‌ی ریاضی پشت آن این است: مشاهده شده که وقتی یک LLM از‌پیش‌آموزش‌دیده روی یک وظیفه‌ی خاص Fine-tune می‌شود، تغییراتی که در ماتریس‌های وزن رخ می‌دهد دارای رتبه‌ی مؤثر پایینی است. به عبارت دیگر، اطلاعات مفیدی که در آموزش اضافه می‌شود در واقع در یک فضای بسیار کوچک‌تر زندگی می‌کند.

LoRA از این مشاهده استفاده می‌کند: به‌جای آن‌که ماتریس وزن W با ابعاد d×d را مستقیماً تغییر دهیم، یک تغییر کوچک ΔW را به‌صورت حاصل‌ضرب دو ماتریس با رتبه‌ی پایین نمایش می‌دهیم:

```
W_new = W_old + ΔW = W_old + B × A
```

در این فرمول، A ماتریسی با ابعاد r×d است و B ماتریسی با ابعاد d×r است که در آن r رتبه‌ی LoRA است و معمولاً مقداری بسیار کوچک مثل ۴، ۸ یا ۱۶ انتخاب می‌شود. اثر این رویکرد روی تعداد پارامترها بسیار چشمگیر است: یک ماتریس ۴۰۹۶×۴۰۹۶ دارای ۱۶.۸ میلیون پارامتر است، اما با LoRA و r=8 فقط ۲×۴۰۹۶×۸ = ۶۵۵۳۶ پارامتر داریم که ۲۵۶ برابر کمتر است. در طول Fine-tuning، W_old کاملاً ثابت نگه‌داشته می‌شود و فقط A و B آموزش می‌بینند.

📷 [تصویر اینجا قرار گیرد: نمودار یک لایه‌ی LoRA که نشان می‌دهد ورودی از دو مسیر موازی عبور می‌کند: مسیر اصلی W که ثابت است و مسیر LoRA که B×A را محاسبه می‌کند و خروجی دو مسیر با هم جمع می‌شوند]

---

# ⚡ QLoRA: ترکیب کوانتیزاسیون و LoRA

QLoRA که در سال ۲۰۲۳ معرفی شد، یک گام فراتر از LoRA می‌رود. علاوه بر استفاده از LoRA برای کاهش پارامترهای قابل‌یادگیری، مدل پایه را هم کوانتیزه می‌کند؛ یعنی وزن‌های مدل که معمولاً با دقت ۱۶ بیت ذخیره می‌شوند، با دقت ۴ بیت ذخیره می‌شوند. این کاهش دقت از ۱۶ بیت به ۴ بیت حافظه‌ی موردنیاز را به یک‌چهارم کاهش می‌دهد. با QLoRA می‌توان یک مدل هفت‌میلیارد پارامتری را روی یک کارت گرافیک با فقط ۱۶ گیگابایت VRAM Fine-tune کرد؛ کاری که قبلاً به ده‌ها گیگابایت VRAM نیاز داشت.

---

# 📋 Instruction Tuning: آموزش پیروی از دستورالعمل

یکی از مهم‌ترین تکنیک‌های Fine-tuning برای LLM‌ها، Instruction Tuning است. یک LLM که فقط با پیش‌آموزش زبانی آموزش دیده، به دستورالعمل‌های انسانی به‌خوبی پاسخ نمی‌دهد. در Instruction Tuning، مدل روی مجموعه‌ای از جفت‌های دستورالعمل-پاسخ Fine-tune می‌شود. هر نمونه یک دستورالعمل (مثل «خلاصه‌ای از این متن بنویس»)، یک ورودی اختیاری و یک پاسخ هدف دارد. پس از این Fine-tuning، مدل یاد می‌گیرد به دستورالعمل‌های انسانی به‌شکل مفید پاسخ دهد.

---

# 💻 پروژه‌ی عملی: پیاده‌سازی دستی LoRA و Fine-tuning با PEFT

```python
import torch
import torch.nn as nn
import numpy as np

# ---- بخش اول: پیاده‌سازی دستی LoRA برای درک مفهوم ----
print("=== پیاده‌سازی دستی LoRA ===\n")

class LoRALayer(nn.Module):
    """
    یک لایه‌ی Linear با افزودن آداپتور LoRA.
    وزن اصلی W ثابت است و فقط ماتریس‌های A و B آموزش می‌بینند.
    """

    def __init__(self, in_features, out_features, rank=4, alpha=16):
        super().__init__()

        # وزن اصلی که در Fine-tuning تغییر نمی‌کند
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.W.weight.requires_grad = False

        # ماتریس‌های LoRA با رتبه‌ی پایین
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # ضریب مقیاس‌بندی
        self.scaling = alpha / rank

        # مقداردهی اولیه: A با توزیع نرمال، B با صفر
        # صفر کردن B تضمین می‌کند که در ابتدا ΔW=0 است
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.W(x) + self.lora_B(self.lora_A(x)) * self.scaling

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# مقایسه‌ی تعداد پارامتر
for d, r in [(256, 4), (512, 8), (1024, 16)]:
    regular = nn.Linear(d, d)
    lora = LoRALayer(d, d, rank=r)
    total_r = sum(p.numel() for p in regular.parameters())
    _, train_l = lora.count_parameters()
    reduction = 100 * (1 - train_l / total_r)
    print(f"بُعد={d}, rank={r}: "
          f"پارامتر کامل={total_r:,} | "
          f"پارامتر LoRA={train_l:,} | "
          f"کاهش={reduction:.1f}%")


# ---- بخش دوم: ساخت یک مدل ساده با LoRA ----
class SimpleTransformerWithLoRA(nn.Module):
    """
    یک مدل ترنسفورمر ساده که لایه‌های Linear آن با LoRA جایگزین شده‌اند
    """

    def __init__(self, vocab_size, d_model=64, num_heads=4, num_layers=2,
                 lora_rank=4, num_classes=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        # لایه‌های LoRA به‌جای Linear معمولی
        self.lora_layers = nn.ModuleList([
            LoRALayer(d_model, d_model, rank=lora_rank)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        out = self.embedding(x)
        for layer in self.lora_layers:
            out = torch.relu(layer(out))
        out = self.norm(out.mean(dim=1))
        return self.classifier(out)


# ---- بخش سوم: Fine-tuning با کتابخانه‌ی PEFT ----
print("\n=== Fine-tuning با PEFT (کد نمایشی) ===\n")

peft_example_code = '''
# نصب: pip install transformers peft datasets accelerate

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# ---- گام اول: بارگذاری مدل پایه ----
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"پارامترهای مدل پایه: {sum(p.numel() for p in base_model.parameters()):,}")

# ---- گام دوم: پیکربندی LoRA ----
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # رتبه‌ی LoRA
    lora_alpha=16,          # ضریب مقیاس‌بندی
    lora_dropout=0.1,       # Dropout برای جلوگیری از بیش‌برازش
    target_modules=["c_attn"],  # ماتریس‌های توجه GPT-2
    bias="none"
)

# اعمال LoRA روی مدل
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# خروجی: trainable params: 294,912 || all params: 124,734,720 || trainable%: 0.24

# ---- گام سوم: دیتاست Instruction Tuning ----
# فرمت Alpaca: Instruction + Response
def format_instruction(instruction, response):
    return f"""### Instruction:
{instruction}

### Response:
{response}{tokenizer.eos_token}"""

qa_data = [
    {
        "instruction": "What is a neural network?",
        "response": "A neural network is a computational model inspired by the "
                   "human brain, consisting of interconnected nodes organized "
                   "in layers that process and learn from data."
    },
    {
        "instruction": "Explain backpropagation.",
        "response": "Backpropagation is the algorithm used to train neural "
                   "networks by computing gradients of the loss function with "
                   "respect to each weight using the chain rule."
    },
    {
        "instruction": "What is overfitting?",
        "response": "Overfitting occurs when a model learns training data too "
                   "well but performs poorly on unseen data. Prevention "
                   "methods include dropout, regularization, and early stopping."
    },
    {
        "instruction": "Describe the transformer architecture.",
        "response": "The transformer uses self-attention mechanisms to process "
                   "sequences in parallel. It consists of encoder and decoder "
                   "stacks with multi-head attention and feed-forward layers."
    },
] * 25  # تکرار برای داده‌ی کافی

# ---- گام چهارم: آموزش ----
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import torch

class InstructionDataset(Dataset):

    def __init__(self, data, tokenizer, max_len=128):
        self.samples = []
        for item in data:
            text = format_instruction(item["instruction"], item["response"])
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt"
            )
            ids = encoded["input_ids"].squeeze()
            self.samples.append({
                "input_ids": ids,
                "attention_mask": encoded["attention_mask"].squeeze(),
                "labels": ids.clone()
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

dataset = InstructionDataset(qa_data, tokenizer)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4,
    weight_decay=0.01
)
num_epochs = 5
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(loader) // 2,
    num_training_steps=len(loader) * num_epochs
)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg = total_loss / len(loader)
    import math
    print(f"دوره {epoch+1} | خطا: {avg:.4f} | Perplexity: {math.exp(avg):.2f}")

# ---- گام پنجم: تولید پاسخ و ذخیره ----
def generate(model, tokenizer, instruction, max_new_tokens=80, device="cpu"):
    prompt = f"### Instruction:\\n{instruction}\\n\\n### Response:\\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return full.split("### Response:")[-1].strip()

print("\\nسؤال: What is deep learning?")
print("پاسخ:", generate(model, tokenizer, "What is deep learning?", device=device))

# ذخیره‌ی فقط آداپتورهای LoRA (نه کل مدل)
model.save_pretrained("./gpt2_lora_adapter")
tokenizer.save_pretrained("./gpt2_lora_adapter")
print("\\nآداپتورهای LoRA ذخیره شدند. (فقط چند مگابایت به‌جای چند صد مگابایت)")
'''

print(peft_example_code)

# ---- مقایسه‌ی روش‌ها ----
print("\n=== مقایسه‌ی روش‌های Fine-tuning ===\n")
print(f"{'روش':<30} {'پارامتر آموزشی':<20} {'حافظه VRAM':<15} {'کیفیت'}")
print("-" * 80)
print(f"{'Full Fine-tuning':<30} {'100٪':<20} {'بسیار زیاد':<15} {'بهترین'}")
print(f"{'LoRA (r=8)':<30} {'~0.1-1٪':<20} {'کم':<15} {'خیلی خوب'}")
print(f"{'QLoRA (4bit + r=8)':<30} {'~0.1-1٪':<20} {'بسیار کم':<15} {'خوب'}")
print(f"{'Freeze + فقط Head':<30} {'<1٪':<20} {'کم':<15} {'متوسط'}")
print()
print("نتیجه: LoRA و QLoRA بهترین تعادل میان کیفیت و مصرف منابع را دارند.")
```

---

# ❓ سؤالات تستی

### سؤال ۱

در LoRA، چرا ماتریس B در ابتدا با مقدار صفر مقداردهی می‌شود؟

الف) برای کاهش سرعت آموزش

ب) تا در ابتدای Fine-tuning تغییر ΔW = B×A برابر صفر باشد و مدل دقیقاً از همان نقطه‌ای که مدل از‌پیش‌آموزش‌دیده بود شروع کند

ج) چون ماتریس‌های صفر محاسبه‌ی سریع‌تری دارند

د) چون ماتریس A همیشه صفر است

### سؤال ۲

تفاوت اصلی QLoRA از LoRA معمولی در چیست؟

الف) QLoRA از رتبه‌ی بالاتری برای ماتریس‌های A و B استفاده می‌کند

ب) QLoRA علاوه بر LoRA، مدل پایه را هم با دقت ۴ بیت کوانتیزه می‌کند که نیاز به حافظه را به‌طور اساسی‌تری کاهش می‌دهد

ج) QLoRA فقط برای مدل‌های کوچک مناسب است

د) QLoRA از معماری متفاوتی نسبت به ترنسفورمر استفاده می‌کند

---

## ✅ پاسخ‌ها

**پاسخ سؤال ۱: گزینه ب**

صفر کردن B تضمین می‌کند که در ابتدای Fine-tuning، تغییر ΔW = B×A = صفر × A = صفر است. این یعنی مدل دقیقاً از همان وضعیت مدل از‌پیش‌آموزش‌دیده شروع می‌کند و به‌تدریج تنها تغییرات لازم را یاد می‌گیرد.

**پاسخ سؤال ۲: گزینه ب**

LoRA معمولی فقط تعداد پارامترهای قابل‌یادگیری را کاهش می‌دهد اما مدل پایه با دقت کامل در حافظه نگه‌داشته می‌شود. QLoRA با کوانتیزه‌کردن مدل پایه به ۴ بیت، حافظه‌ی موردنیاز برای وزن‌های پایه را هم به یک‌چهارم کاهش می‌دهد که Fine-tuning مدل‌های بسیار بزرگ روی سخت‌افزار معمولی را ممکن می‌سازد.

---

# 📝 خلاصه فصل

در این فصل با چالش‌های اصلی Fine-tuning کامل LLM‌ها آشنا شدیم و دیدیم که نیاز به ده‌ها گیگابایت VRAM این کار را برای اکثر توسعه‌دهندگان غیرممکن می‌کند. LoRA با تقریب تغییرات وزن به ضرب دو ماتریس با رتبه‌ی پایین، پارامترهای قابل‌یادگیری را تا ۹۹ درصد کاهش می‌دهد. QLoRA با ترکیب این ایده با کوانتیزاسیون ۴ بیتی، Fine-tuning مدل‌های بسیار بزرگ را حتی روی یک GPU مصرف‌کننده ممکن کرده است. Instruction Tuning هم مدل‌ها را برای پیروی از دستورالعمل‌های انسانی آماده می‌کند. در پروژه‌ی عملی هم پیاده‌سازی دستی LoRA را دیدیم و هم کد کامل Fine-tuning با کتابخانه‌ی PEFT را بررسی کردیم.

---

# 🎯 تمرین‌های پایان فصل

۱. رتبه‌ی LoRA را از ۸ به ۴ و ۱۶ تغییر دهید و تأثیر آن بر کیفیت پاسخ و تعداد پارامتر را مقایسه کنید.
۲. به‌جای `c_attn`، LoRA را به `c_proj` هم اضافه کنید و نتایج را مقایسه کنید.
۳. یک دیتاست اختصاصی در حوزه‌ی مورد علاقه‌تان به فرمت Instruction-Response بسازید.
۴. مدل Fine-tune شده را با مدل پایه مقایسه کنید.
۵. با نصب `bitsandbytes`، QLoRA را با `load_in_4bit=True` امتحان کنید.

---

در فصل ۲۵ با **معماری RAG (Retrieval-Augmented Generation)** آشنا می‌شویم که با ترکیب بازیابی اطلاعات و تولید متن، محدودیت دانش ثابت LLM‌ها را برطرف می‌کند.
