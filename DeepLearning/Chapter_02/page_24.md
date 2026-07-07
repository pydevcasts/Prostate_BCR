# فصل ۲۴ 🔧 Fine-tuning پیشرفته‌ی مدل‌های زبانی بزرگ: LoRA و QLoRA

## 🎯 اهداف فصل

در فصل قبل Fine-tuning استاندارد BERT را یاد گرفتیم که تمام پارامترهای مدل در آموزش شرکت می‌کنند. اما برای مدل‌های زبانی بزرگ‌تر مثل LLaMA-2 با هفت میلیارد پارامتر یا GPT-3 با ۱۷۵ میلیارد پارامتر، این رویکرد نیاز به حافظه‌ی گرافیکی (VRAM) بسیار زیادی دارد که برای اکثر توسعه‌دهندگان در دسترس نیست. در این فصل تکنیک‌های PEFT (Parameter-Efficient Fine-Tuning) را یاد می‌گیریم که آموزش مدل‌های بسیار بزرگ را حتی با منابع محدود ممکن می‌سازند. در پایان این فصل خواهید توانست مشکل اصلی Fine-tuning کامل مدل‌های بزرگ را توضیح دهید، مفهوم LoRA (Low-Rank Adaptation) را با استدلال ریاضی درک کنید، تفاوت LoRA و QLoRA را بشناسید، مفهوم Instruction Tuning را توضیح دهید، و یک مدل LLM کوچک را با LoRA روی یک دیتاست اختصاصی Fine-tune کنید.

---

# 💸 چالش Fine-tuning کامل مدل‌های بزرگ

وقتی بخواهیم یک مدل با هفت میلیارد پارامتر را Fine-tune کنیم، تنها نگه‌داشتن وزن‌های آن در حافظه به ۱۴ گیگابایت (با دقت FP16) نیاز دارد. اما در طول آموزش، علاوه بر وزن‌های مدل، باید گرادیان‌ها (۱۴ گیگابایت دیگر) و وضعیت بهینه‌ساز Adam (۲۸ گیگابایت دیگر برای ذخیره‌ی دو لحظه‌ی گرادیان) هم در حافظه نگه‌داشته شوند. این یعنی فقط برای آموزش یک مدل هفت‌میلیارد پارامتری به حدود ۵۶ گیگابایت VRAM نیاز است که حتی بهترین کارت‌های گرافیک مصرف‌کننده‌ای هم آن را ندارند.

این واقعیت باعث می‌شود که تنها شرکت‌های بزرگ با زیرساخت GPU گران‌قیمت بتوانند LLM بسازند یا Fine-tune کنند. اما تکنیک‌های PEFT این معادله را تغییر داده‌اند و آموزش یا تنظیم مدل‌های بزرگ را با منابع بسیار محدودتر ممکن کرده‌اند.

---

# 📐 LoRA: تقریب ماتریس وزن با رتبه‌ی پایین

LoRA که مخفف Low-Rank Adaptation است، در سال ۲۰۲۱ توسط هو و همکارانش معرفی شد. ایده‌ی ریاضی پشت آن ساده اما قدرتمند است. مشاهده شده که وقتی یک LLM از‌پیش‌آموزش‌دیده روی یک وظیفه‌ی خاص Fine-tune می‌شود، تغییراتی که در ماتریس‌های وزن رخ می‌دهد دارای رتبه‌ی مؤثر پایینی است. به عبارت دیگر، اطلاعات مفیدی که در آموزش اضافه می‌شود در واقع در یک فضای بسیار کوچک‌تر زندگی می‌کند.

LoRA از این مشاهده استفاده می‌کند: به‌جای آن‌که ماتریس وزن W (با ابعاد d×d) را مستقیماً تغییر دهیم، یک تغییر کوچک ΔW را به‌صورت حاصل‌ضرب دو ماتریس با رتبه‌ی پایین نمایش می‌دهیم:

```
W_new = W_old + ΔW = W_old + B × A
```

در این فرمول، A ماتریسی با ابعاد r×d است و B ماتریسی با ابعاد d×r است که در آن r رتبه‌ی LoRA است و معمولاً مقداری بسیار کوچک مثل ۴، ۸ یا ۱۶ انتخاب می‌شود. حاصل‌ضرب B×A یک ماتریس d×d است که همان تغییر ΔW را تقریب می‌زند.

اثر این رویکرد روی تعداد پارامترها بسیار چشمگیر است. یک ماتریس d×d با d=4096 دارای ۴۰۹۶×۴۰۹۶ = ۱۶.۸ میلیون پارامتر است. اما با LoRA و r=8، ماتریس‌های A و B جمعاً فقط ۲×۴۰۹۶×۸ = ۶۵۵۳۶ پارامتر دارند که تقریباً ۲۵۶ برابر کمتر است. در طول Fine-tuning، W_old کاملاً ثابت نگه‌داشته می‌شود و فقط A و B آموزش می‌بینند که نیاز به حافظه را به‌طور اساسی کاهش می‌دهد.

📷 [تصویر اینجا قرار گیرد: نمودار یک لایه‌ی LoRA که نشان می‌دهد ورودی از دو مسیر موازی عبور می‌کند: مسیر اصلی W که ثابت است و مسیر LoRA که B×A را محاسبه می‌کند و خروجی دو مسیر جمع می‌شوند]

---

# ⚡ QLoRA: ترکیب کوانتیزاسیون و LoRA

QLoRA که در سال ۲۰۲۳ معرفی شد، یک گام فراتر از LoRA می‌رود. علاوه بر استفاده از LoRA برای کاهش پارامترهای قابل‌یادگیری، مدل پایه را هم کوانتیزه (Quantize) می‌کند؛ یعنی وزن‌های مدل که معمولاً با دقت ۳۲ بیت (FP32) یا ۱۶ بیت (FP16) ذخیره می‌شوند، با دقت ۴ بیت (NF4) ذخیره می‌شوند. این کاهش دقت از ۱۶ بیت به ۴ بیت، حافظه‌ی موردنیاز را به یک‌چهارم کاهش می‌دهد.

با QLoRA می‌توان یک مدل هفت‌میلیارد پارامتری را روی یک کارت گرافیک مصرف‌کننده‌ای با فقط ۱۶ گیگابایت VRAM Fine-tune کرد؛ کاری که قبلاً به ده‌ها گیگابایت VRAM در چندین GPU نیاز داشت. این نوآوری Fine-tuning LLM را برای محققان مستقل، استارتاپ‌های کوچک و حتی توسعه‌دهندگان فردی در دسترس قرار داده است.

---

# 📋 Instruction Tuning: آموزش پیروی از دستورالعمل

یکی از مهم‌ترین تکنیک‌های Fine-tuning برای مدل‌های زبانی بزرگ، Instruction Tuning است. یک LLM که فقط با پیش‌آموزش زبانی (پیش‌بینی کلمه‌ی بعدی) آموزش دیده، به دستورالعمل‌های انسانی به‌خوبی پاسخ نمی‌دهد. به‌عنوان مثال اگر از آن بپرسید «سه مزیت یادگیری عمیق را بنویس»، ممکن است فقط متن مشابه‌ای ادامه دهد بدون آن‌که واقعاً پاسخ مفیدی بدهد.

در Instruction Tuning، مدل روی مجموعه‌ای از جفت‌های دستورالعمل-پاسخ Fine-tune می‌شود. هر نمونه از آموزش یک دستورالعمل (مثل «خلاصه‌ای از این متن بنویس»)، یک ورودی اختیاری (متن برای خلاصه‌کردن) و یک پاسخ هدف دارد. پس از این Fine-tuning، مدل یاد می‌گیرد که به دستورالعمل‌های انسانی به‌شکل مفید پاسخ دهد.

---

# 💻 پروژه‌ی عملی فصل: Fine-tuning با LoRA روی دیتاست اختصاصی

در این پروژه از کتابخانه‌ی PEFT از Hugging Face برای Fine-tuning یک مدل GPT-2 کوچک با LoRA روی یک دیتاست پرسش‌وپاسخ اختصاصی استفاده می‌کنیم. این پروژه دقیقاً همان الگویی است که برای مدل‌های بزرگ‌تر هم به کار می‌رود.

```python
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# ---- نصب کتابخانه‌های موردنیاز ----
# pip install transformers peft datasets accelerate bitsandbytes

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        get_cosine_schedule_with_warmup
    )
    from peft import (
        LoraConfig, get_peft_model,
        TaskType, PeftModel
    )
    PEFT_AVAILABLE = True
    print("PEFT و Transformers با موفقیت بارگذاری شدند.")
except ImportError:
    PEFT_AVAILABLE = False
    print("کتابخانه‌های PEFT نصب نیستند.")
    print("نصب: pip install transformers peft accelerate")


# ---- نمایش مفهوم LoRA با پیاده‌سازی دستی ----
print("\n=== پیاده‌سازی دستی LoRA برای درک مفهوم ===\n")

class LoRALayer(nn.Module):
    """
    یک لایه‌ی Linear ساده با LoRA.
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

        # ضریب مقیاس‌بندی: alpha/rank
        self.scaling = alpha / rank

        # مقداردهی اولیه: A با توزیع نرمال، B با صفر
        # صفر کردن B تضمین می‌کند که در ابتدا ΔW=0 است
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # خروجی = Wx + (scaling × B×A×x)
        base_output = self.W(x)
        lora_output = self.lora_B(self.lora_A(x)) * self.scaling
        return base_output + lora_output

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# مقایسه‌ی تعداد پارامتر
d = 1024
r = 8

regular_layer = nn.Linear(d, d)
lora_layer = LoRALayer(d, d, rank=r)

total_lora, trainable_lora = lora_layer.count_parameters()
total_regular = sum(p.numel() for p in regular_layer.parameters())

print(f"لایه‌ی Linear معمولی ({d}×{d}):")
print(f"  پارامتر قابل‌یادگیری: {total_regular:,}")

print(f"\nلایه‌ی LoRA ({d}×{d}, rank={r}):")
print(f"  کل پارامتر: {total_lora:,}")
print(f"  پارامتر قابل‌یادگیری: {trainable_lora:,}")
print(f"  کاهش نسبت به روش کامل: {100*(1 - trainable_lora/total_regular):.1f}%")


if PEFT_AVAILABLE:

    # ---- بارگذاری مدل پایه ----
    print("\n=== Fine-tuning GPT-2 با LoRA ===\n")

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # بارگذاری مدل پایه
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )

    print(f"پارامترهای مدل پایه: "
          f"{sum(p.numel() for p in base_model.parameters()):,}")

    # ---- پیکربندی LoRA ----
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,

        # رتبه‌ی LoRA: عدد کوچک‌تر یعنی پارامتر کمتر
        r=8,

        # alpha برای مقیاس‌بندی: معمولاً 2×r یا r
        lora_alpha=16,

        # Dropout برای جلوگیری از بیش‌برازش
        lora_dropout=0.1,

        # کدام ماتریس‌های وزن LoRA اضافه شود
        # در GPT-2: ماتریس‌های توجه (q_proj, v_proj)
        target_modules=["c_attn"],

        bias="none"
    )

    # اعمال LoRA روی مدل
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # ---- دیتاست Instruction Tuning ----
    # داده‌ی نمونه برای آموزش مدل پرسش‌وپاسخ در حوزه‌ی یادگیری عمیق
    qa_data = [
        {
            "instruction": "What is a neural network?",
            "response": "A neural network is a machine learning model inspired by "
                       "the human brain, consisting of layers of interconnected "
                       "nodes that process information."
        },
        {
            "instruction": "Explain backpropagation in simple terms.",
            "response": "Backpropagation is the algorithm that trains neural "
                       "networks by calculating how much each weight contributed "
                       "to the error and adjusting them accordingly."
        },
        {
            "instruction": "What is the difference between CNN and RNN?",
            "response": "CNNs are designed for spatial data like images using "
                       "convolutional filters, while RNNs handle sequential data "
                       "like text using recurrent connections to maintain memory."
        },
        {
            "instruction": "What is overfitting and how to prevent it?",
            "response": "Overfitting occurs when a model learns training data too "
                       "well but fails on new data. Prevention methods include "
                       "dropout, regularization, early stopping, and data augmentation."
        },
        {
            "instruction": "Explain the transformer architecture.",
            "response": "Transformers use self-attention mechanisms to process "
                       "sequences in parallel, replacing recurrent networks. "
                       "They consist of encoder and decoder stacks with "
                       "multi-head attention and feed-forward layers."
        },
    ] * 20  # تکرار برای داده‌ی کافی

    # ---- فرمت‌دهی دیتاست به شکل Alpaca ----
    def format_instruction(sample):
        return f"""### Instruction:
{sample['instruction']}

### Response:
{sample['response']}{tokenizer.eos_token}"""

    class InstructionDataset(Dataset):

        def __init__(self, data, tokenizer, max_len=256):
            self.samples = []
            for item in data:
                text = format_instruction(item)
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_len,
                    padding='max_length',
                    return_tensors='pt'
                )
                input_ids = encoded['input_ids'].squeeze()
                # برای Causal LM، labels همان input_ids هستند
                self.samples.append({
                    'input_ids': input_ids,
                    'attention_mask': encoded['attention_mask'].squeeze(),
                    'labels': input_ids.clone()
                })

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]


    dataset = InstructionDataset(qa_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # ---- آموزش با LoRA ----
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4,
        weight_decay=0.01
    )

    num_epochs = 5
    total_steps = len(dataloader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    print("\nشروع آموزش با LoRA...\n")

    for epoch in range(num_epochs):

        model.train()
        total_loss = 0.0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        perplexity = np.exp(avg_loss)
        print(f"دوره {epoch+1} | خطا: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")

    # ---- آزمایش مدل Fine-tune شده ----
    def generate_response(model, tokenizer, instruction, max_new_tokens=100):
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("### Response:")[-1].strip()


    print("\n=== آزمایش مدل Fine-tune شده ===\n")
    test_questions = [
        "What is deep learning?",
        "How does LSTM solve the vanishing gradient problem?"
    ]

    for question in test_questions:
        print(f"سؤال: {question}")
        answer = generate_response(model, tokenizer, question)
        print(f"پاسخ: {answer}\n")

    # ---- ذخیره‌ی فقط آداپتورهای LoRA (نه کل مدل) ----
    model.save_pretrained('./gpt2_lora_qa')
    tokenizer.save_pretrained('./gpt2_lora_qa')
    print("آداپتورهای LoRA ذخیره شدند در: ./gpt2_lora_qa")
    print("(فقط وزن‌های LoRA ذخیره می‌شوند، نه کل مدل)")


# ---- مقایسه‌ی روش‌های Fine-tuning ----
print("\n=== مقایسه‌ی روش‌های مختلف Fine-tuning ===\n")
print(f"{'روش':<25} {'پارامتر آموزشی':<20} {'حافظه VRAM':<15} {'کیفیت'}")
print("-" * 75)
print(f"{'Full Fine-tuning':<25} {'100%':<20} {'بسیار زیاد':<15} {'بهترین'}")
print(f"{'LoRA (r=8)':<25} {'~0.1-1%':<20} {'کم':<15} {'خیلی خوب'}")
print(f"{'QLoRA (4bit+r=8)':<25} {'~0.1-1%':<20} {'بسیار کم':<15} {'خوب'}")
print(f"{'Freeze + Head Only':<25} {'<1%':<20} {'کم':<15} {'متوسط'}")
```

---

# ❓ سؤالات تستی

### سؤال ۱

در LoRA، چرا ماتریس B در ابتدا با مقدار صفر مقداردهی می‌شود؟

الف) برای کاهش سرعت آموزش

ب) تا در ابتدای Fine-tuning، تغییر ΔW = B×A برابر صفر باشد و مدل دقیقاً از همان نقطه‌ای شروع کند که مدل از‌پیش‌آموزش‌دیده بود

ج) چون ماتریس‌های صفر محاسبه‌ی سریع‌تری دارند

د) چون ماتریس A همیشه صفر است

### سؤال ۲

تفاوت اصلی QLoRA از LoRA معمولی در چیست؟

الف) QLoRA از رتبه‌ی بالاتری برای ماتریس‌های A و B استفاده می‌کند

ب) QLoRA علاوه بر LoRA، مدل پایه را هم کوانتیزه می‌کند (مثلاً با دقت ۴ بیت) که نیاز به حافظه را به‌طور اساسی‌تری کاهش می‌دهد

ج) QLoRA فقط برای مدل‌های کوچک مناسب است

د) QLoRA از معماری متفاوتی نسبت به ترنسفورمر استفاده می‌کند

---

## ✅ پاسخ‌ها

**پاسخ سؤال ۱: گزینه ب**

صفر کردن B تضمین می‌کند که در ابتدای Fine-tuning، تغییر ΔW = B×A = صفر × A = صفر است. این یعنی مدل دقیقاً از همان وضعیت مدل از‌پیش‌آموزش‌دیده شروع می‌کند و به‌تدریج تنها تغییرات لازم را یاد می‌گیرد. اگر B با مقادیر تصادفی مقداردهی می‌شد، ΔW اولیه یک اختلال تصادفی به مدل وارد می‌کرد که آموزش را بی‌ثبات‌تر می‌کرد.

**پاسخ سؤال ۲: گزینه ب**

LoRA معمولی فقط تعداد پارامترهای قابل‌یادگیری را کاهش می‌دهد اما مدل پایه همچنان با دقت کامل (FP16 یا FP32) در حافظه نگه‌داشته می‌شود. QLoRA با کوانتیزه‌کردن مدل پایه به ۴ بیت، حافظه‌ی موردنیاز برای نگه‌داری وزن‌های پایه را هم به یک‌چهارم کاهش می‌دهد که Fine-tuning مدل‌های بسیار بزرگ روی سخت‌افزار معمولی را ممکن می‌سازد.

---

# 📝 خلاصه فصل

در این فصل با چالش‌های اصلی Fine-tuning کامل مدل‌های زبانی بزرگ آشنا شدیم و دیدیم که نیاز به ده‌ها گیگابایت VRAM این کار را برای اکثر توسعه‌دهندگان غیرممکن می‌کند. LoRA با تقریب تغییرات وزن به ضرب دو ماتریس با رتبه‌ی پایین، پارامترهای قابل‌یادگیری را تا ۹۹ درصد کاهش می‌دهد. QLoRA با ترکیب این ایده با کوانتیزاسیون ۴ بیتی مدل پایه، Fine-tuning مدل‌های بسیار بزرگ را حتی روی یک GPU مصرف‌کننده‌ای ممکن کرده است. مفهوم Instruction Tuning را هم یاد گرفتیم که مدل‌های زبانی را برای پیروی از دستورالعمل‌های انسانی آموزش می‌دهد. در پروژه‌ی عملی یک Fine-tuning کامل با LoRA روی GPT-2 پیاده‌سازی کردیم.

---

# 🎯 تمرین‌های پایان فصل

۱. رتبه‌ی LoRA را از ۸ به ۴ و ۱۶ تغییر دهید و تأثیر آن بر کیفیت پاسخ و تعداد پارامتر را مقایسه کنید.

۲. به‌جای `c_attn`، LoRA را به `c_proj` هم اضافه کنید و نتایج را مقایسه کنید.

۳. یک دیتاست اختصاصی در حوزه‌ی مورد علاقه‌تان (مثل پزشکی، حقوق یا آموزش) به فرمت Instruction-Response بسازید و مدل را روی آن Fine-tune کنید.

۴. مدل Fine-tune شده را با مدل پایه مقایسه کنید: هر دو را با همان سؤال‌ها آزمایش کنید و ببینید آموزش چقدر پاسخ‌ها را بهتر کرده.

۵. کتابخانه‌ی `bitsandbytes` را نصب کنید و QLoRA را با `load_in_4bit=True` روی یک مدل بزرگ‌تر مثل GPT-2 Medium امتحان کنید.

---

در فصل ۲۵ با **معماری RAG (Retrieval-Augmented Generation)** آشنا می‌شویم؛ رویکردی که با ترکیب بازیابی اطلاعات و تولید متن، محدودیت دانش ثابت مدل‌های زبانی را برطرف می‌کند و یک سامانه‌ی پرسش‌وپاسخ بر اساس اسناد واقعی می‌سازد.