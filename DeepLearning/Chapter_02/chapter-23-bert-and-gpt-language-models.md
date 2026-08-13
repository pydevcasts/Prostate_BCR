# فصل ۲۳ 🤖 مدل‌های زبانی مبتنی بر ترنسفورمر: BERT، GPT و Hugging Face

## 🎯 اهداف فصل

در فصل‌های قبل معماری ترنسفورمر را از صفر پیاده‌سازی کردیم. اکنون می‌خواهیم ببینیم که چگونه این معماری به مدل‌های زبانی بزرگی مثل BERT و GPT تبدیل شد که امروزه در تقریباً هر کاربرد NLP صنعتی حضور دارند. در پایان این فصل خواهید توانست تفاوت ماهوی رویکرد BERT (Encoder-only) و GPT (Decoder-only) را توضیح دهید، مفاهیم Pretraining و Fine-tuning را برای مدل‌های زبانی بزرگ درک کنید، از کتابخانه‌ی Hugging Face Transformers برای بارگذاری، استفاده و Fine-tune کردن مدل‌های از‌پیش‌آموزش‌دیده روی وظایف مختلف NLP استفاده کنید، و تفاوت Tokenizer مدل‌های ترنسفورمری مثل WordPiece و BPE را با توکنایزرهای ساده‌تر درک کنید.

---

# 🧠 BERT: خواندن متن از هر دو طرف

BERT که مخفف Bidirectional Encoder Representations from Transformers است، در سال ۲۰۱۸ توسط گوگل معرفی شد و نتایج حیرت‌انگیزی در بنچمارک‌های مختلف NLP ثبت کرد. ایده‌ی اصلی BERT استفاده از تمام لایه‌های Encoder ترنسفورمر برای ساخت نمایش‌های غنی از زبان است. کلمه‌ی «Bidirectional» در نام آن اشاره به این دارد که BERT برخلاف مدل‌های قبلی، هنگام پردازش هر کلمه به هر دوی چپ و راست آن نگاه می‌کند. این یعنی درک کلمه‌ی «بانک» در «بانک رودخانه» با درک آن در «بانک پول» کاملاً متفاوت خواهد بود.

BERT با دو وظیفه‌ی پیش‌آموزش طراحی شده است. وظیفه‌ی اول Masked Language Model یا MLM است که در آن حدود پانزده درصد از کلمات متن تصادفاً با یک توکن `[MASK]` جایگزین می‌شوند و مدل باید کلمه‌ی اصلی را پیش‌بینی کند. این وظیفه مدل را مجبور می‌کند از زمینه‌ی اطراف هر کلمه برای درک معنای آن استفاده کند. وظیفه‌ی دوم Next Sentence Prediction یا NSP است که در آن مدل باید تشخیص دهد آیا دو جمله واقعاً پشت‌سرهم در متن آمده‌اند یا به‌طور تصادفی کنار هم گذاشته شده‌اند. این وظیفه مدل را برای درک روابط میان جملات آموزش می‌دهد.

📷 [تصویر اینجا قرار گیرد: نمودار معماری BERT شامل پشته‌ای از N لایه Encoder با نمایش ورودی‌های ویژه CLS، SEP و MASK، و خروجی‌های نمایش هر توکن]

---

# 📖 GPT: تولید متن از چپ به راست

GPT که مخفف Generative Pre-trained Transformer است، رویکردی مخالف BERT دارد. GPT از لایه‌های Decoder ترنسفورمر استفاده می‌کند و به‌صورت یک‌طرفه (از چپ به راست) متن را پردازش می‌کند. در هر گام، GPT فقط به کلمات قبل از کلمه‌ی فعلی دسترسی دارد (همان Causal Mask که در فصل ۲۰ دیدیم) و وظیفه‌ی پیش‌آموزش آن پیش‌بینی کلمه‌ی بعدی است.

این طراحی GPT را برای وظایف تولید متن (Text Generation) بسیار مناسب می‌کند، چراکه ذاتاً به‌صورت چپ‌به‌راست متن تولید می‌کند. خانواده‌ی GPT از نسل اول با ۱۱۷ میلیون پارامتر شروع شد و به GPT-4 با صدها میلیارد پارامتر رسیده است. مدل‌های GPT پایه‌ی ChatGPT هستند که با Fine-tuning از بازخورد انسانی (RLHF) برای مکالمه بهینه شده‌اند.

---

# 🔑 تفاوت ماهوی BERT و GPT

| ویژگی | BERT | GPT |
|---|---|---|
| نوع معماری | Encoder-only | Decoder-only |
| جهت توجه | دوطرفه | یک‌طرفه (چپ به راست) |
| وظیفه‌ی پیش‌آموزش | پیش‌بینی کلمه‌ی ماسک‌شده | پیش‌بینی کلمه‌ی بعدی |
| قوی‌تر در | درک و طبقه‌بندی متن | تولید متن |
| نمونه‌های مشهور | BERT، RoBERTa، ALBERT | GPT-2، GPT-3، GPT-4 |

---

# 🤗 کتابخانه‌ی Hugging Face Transformers

کتابخانه‌ی Hugging Face Transformers یکی از ارزشمندترین ابزارهای موجود در اکوسیستم یادگیری عمیق است که دسترسی به صدها مدل از‌پیش‌آموزش‌دیده را با چند خط کد ممکن می‌کند. این کتابخانه از هم PyTorch و هم TensorFlow پشتیبانی می‌کند و دو مفهوم اصلی دارد: Tokenizer که متن خام را به اندیس‌هایی که مدل می‌فهمد تبدیل می‌کند، و Model که معماری واقعی ترنسفورمر است. این دو همیشه باید با هم از همان نقطه‌ی بارگذاری شوند تا سازگار باشند.

Tokenizer مدل‌های مدرن مثل WordPiece (در BERT) یا BPE (در GPT) از Subword Tokenization استفاده می‌کنند که در فصل ۲۱ با مفهوم کلی آن آشنا شدیم. یک مثال ساده نشان می‌دهد که کلمه‌ی «unbelievable» ممکن است به توکن‌های «un»، «##believ» و «##able» تقسیم شود، جایی که پیشوند «##» نشان می‌دهد این تکه ادامه‌ی کلمه‌ی قبلی است نه کلمه‌ی جدید.

---

# 💻 پروژه‌ی عملی فصل: Fine-tuning BERT برای تحلیل احساسات

در این پروژه از مدل BERT از‌پیش‌آموزش‌دیده برای تحلیل احساسات متون روی دیتاست IMDB استفاده می‌کنیم. این دقیقاً همان جریان کاری است که در پروژه‌های صنعتی واقعی استفاده می‌شود.

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import numpy as np

# ---- نصب Hugging Face اگر نصب نیست ----
# pip install transformers datasets

try:
    from transformers import (
        BertTokenizer,
        BertForSequenceClassification,
        get_linear_schedule_with_warmup
    )
    from datasets import load_dataset
    HF_AVAILABLE = True
    print("Hugging Face Transformers با موفقیت بارگذاری شد.")
except ImportError:
    HF_AVAILABLE = False
    print("Hugging Face نصب نیست. ساختار کد را می‌بینید اما اجرا نمی‌شود.")
    print("نصب: pip install transformers datasets")


if HF_AVAILABLE:

    # ---- بارگذاری توکنایزر و مدل BERT ----
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # نمایش کارکرد توکنایزر BERT
    sample_text = "Deep learning is revolutionizing AI!"
    tokens = tokenizer.tokenize(sample_text)
    token_ids = tokenizer.encode(sample_text)
    print(f"\nمتن اصلی: {sample_text}")
    print(f"توکن‌های BERT: {tokens}")
    print(f"اندیس‌های توکن: {token_ids}")
    print(f"توکن‌های ویژه: [CLS]={tokenizer.cls_token_id}, "
          f"[SEP]={tokenizer.sep_token_id}, "
          f"[MASK]={tokenizer.mask_token_id}")

    # ---- بارگذاری دیتاست IMDB ----
    print("\nبارگذاری دیتاست IMDB...")
    dataset = load_dataset("imdb")
    train_data = dataset['train'].select(range(2000))
    test_data = dataset['test'].select(range(500))
    print(f"نمونه آموزشی: {len(train_data)} | آزمون: {len(test_data)}")

    # ---- دیتاست سفارشی ----
    class IMDBDataset(Dataset):

        def __init__(self, data, tokenizer, max_len=128):
            self.texts = data['text']
            self.labels = data['label']
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }


    train_dataset = IMDBDataset(train_data, tokenizer)
    test_dataset = IMDBDataset(test_data, tokenizer)

    trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # ---- بارگذاری مدل BERT برای طبقه‌بندی ----
    print("\nبارگذاری مدل BERT...")
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"کل پارامترها: {n_params_total:,}")
    print(f"پارامترهای قابل‌یادگیری: {n_params_train:,}")

    # ---- تنظیم بهینه‌ساز با نرخ‌های یادگیری متفاوت ----
    # لایه‌های BERT با نرخ کوچک‌تر آموزش می‌بینند (Fine-tuning ملایم)
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': 2e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=0.01)

    num_epochs = 3
    total_steps = len(trainloader) * num_epochs

    # Warmup Scheduler: نرخ یادگیری ابتدا رشد می‌کند سپس کاهش می‌یابد
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    # ---- حلقه‌ی آموزش ----
    print(f"\n=== آموزش BERT روی IMDB ({num_epochs} دوره) ===\n")

    for epoch in range(num_epochs):

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(trainloader):
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
            logits = outputs.logits

            loss.backward()
            # Gradient Clipping برای پایداری آموزش BERT
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 20 == 0:
                print(f"  دوره {epoch+1} | دسته {batch_idx+1}/{len(trainloader)} | "
                      f"خطا: {total_loss/(batch_idx+1):.4f} | "
                      f"دقت: {100*correct/total:.1f}%")

        # ارزیابی دوره
        model.eval()
        test_preds = []
        test_true = []

        with torch.no_grad():
            for batch in testloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                preds = outputs.logits.argmax(1).cpu()
                test_preds.extend(preds.numpy())
                test_true.extend(labels.numpy())

        test_acc = 100 * sum(p == t for p, t in zip(test_preds, test_true)) / len(test_true)
        print(f"\nپایان دوره {epoch+1}: دقت آزمون = {test_acc:.2f}%\n")

    # گزارش نهایی
    print("گزارش کامل ارزیابی:")
    print(classification_report(test_true, test_preds,
                                target_names=['منفی', 'مثبت']))

    # ذخیره مدل Fine-tune شده
    model.save_pretrained('./bert_imdb_model')
    tokenizer.save_pretrained('./bert_imdb_model')
    print("\nمدل Fine-tune شده ذخیره شد در: ./bert_imdb_model")


# ---- نمایش استفاده از Pipeline برای سرعت بیشتر ----
print("\n=== استفاده از Pipeline برای استنتاج سریع ===\n")
pipeline_code = '''
from transformers import pipeline

# استفاده از مدل Fine-tune شده یا مدل آماده‌ی Hugging Face
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="./bert_imdb_model"  # یا مثلاً "distilbert-base-uncased-finetuned-sst-2-english"
)

# تحلیل جملات جدید
texts = [
    "This movie was absolutely fantastic!",
    "I wasted two hours of my life watching this.",
    "The acting was okay but the story was boring."
]

results = sentiment_analyzer(texts)
for text, result in zip(texts, results):
    print(f"متن: {text}")
    print(f"احساس: {result['label']} | اطمینان: {result['score']:.3f}")
    print()
'''
print(pipeline_code)


# ---- بخش تکمیلی: مقایسه‌ی BERT با LSTM ----
print("=== مقایسه‌ی BERT با LSTM (نتایج معمول روی IMDB) ===\n")
print(f"{'مدل':<30} {'دقت معمول':<15} {'پارامتر':<20} {'زمان آموزش'}")
print("-" * 75)
print(f"{'LSTM ساده با GloVe':<30} {'88-90%':<15} {'~5M':<20} {'سریع'}")
print(f"{'BERT-base Fine-tuned':<30} {'93-95%':<15} {'110M':<20} {'کند'}")
print(f"{'DistilBERT Fine-tuned':<30} {'91-93%':<15} {'66M':<20} {'متوسط'}")
print(f"{'RoBERTa Fine-tuned':<30} {'95-96%':<15} {'125M':<20} {'کند'}")
print()
print("نتیجه: BERT دقت بالاتری دارد اما به منابع محاسباتی بیشتری نیاز دارد.")
print("برای پروژه‌های با منابع محدود، DistilBERT تعادل خوبی برقرار می‌کند.")
```

---

# ❓ سؤالات تستی

### سؤال ۱

چرا BERT برای وظایف درک متن (مثل تحلیل احساسات و پرسش‌وپاسخ) مناسب‌تر از GPT است؟

الف) چون BERT پارامترهای بیشتری دارد

ب) چون BERT از معماری Encoder دوطرفه استفاده می‌کند و هنگام پردازش هر کلمه به هر دوی چپ و راست آن دسترسی دارد که درک معنای کلمه در زمینه را بهتر می‌کند

ج) چون GPT فقط روی زبان انگلیسی کار می‌کند

د) چون BERT قدیمی‌تر از GPT است

### سؤال ۲

در Fine-tuning BERT، چرا از نرخ‌های یادگیری متفاوت برای لایه‌های BERT و طبقه‌بند نهایی استفاده می‌شود؟

الف) چون PyTorch از نرخ یادگیری یکسان پشتیبانی نمی‌کند

ب) چون لایه‌های BERT از قبل خوب آموزش دیده‌اند و فقط نیاز به تنظیم ظریف با نرخ یادگیری کوچک دارند، در حالی‌که طبقه‌بند جدید که از صفر شروع می‌کند، نیاز به نرخ یادگیری بزرگ‌تری دارد

ج) چون لایه‌های BERT در حین Fine-tuning نباید تغییر کنند

د) چون نرخ‌های یادگیری متفاوت سرعت محاسبات را افزایش می‌دهند

---

## ✅ پاسخ‌ها

**پاسخ سؤال ۱: گزینه ب**

در Masked Self-Attention مورد استفاده در GPT، هر توکن فقط به توکن‌های قبل از خود توجه می‌کند. اما در BERT که از Encoder کامل استفاده می‌کند، هر توکن به تمام توکن‌های دیگر (هم قبل هم بعد) دسترسی دارد. این دوطرفه‌بودن BERT را برای وظایفی که نیاز به درک کامل معنای جمله دارند بسیار قوی‌تر می‌کند.

**پاسخ سؤال ۲: گزینه ب**

این تکنیک که Discriminative Fine-tuning نام دارد، از اصل Transfer Learning پیروی می‌کند. لایه‌های BERT وزن‌های ارزشمندی از آموزش روی میلیاردها کلمه دارند و تغییر زیاد آن‌ها این دانش را از بین می‌برد. نرخ یادگیری بسیار کوچک‌تر (۲e-5 در برابر ۱e-4) اطمینان می‌دهد که این وزن‌ها فقط به‌آرامی برای وظیفه‌ی جدید تنظیم می‌شوند.

---

# 📝 خلاصه فصل

در این فصل با دو خانواده‌ی اصلی مدل‌های زبانی مبتنی بر ترنسفورمر آشنا شدیم. BERT با معماری Encoder دوطرفه و پیش‌آموزش MLM، برای وظایف درک متن بهترین عملکرد را دارد. GPT با معماری Decoder یک‌طرفه و پیش‌آموزش پیش‌بینی کلمه‌ی بعدی، برای تولید متن مناسب‌تر است. کتابخانه‌ی Hugging Face Transformers را به‌عنوان ابزار اصلی صنعتی برای کار با این مدل‌ها شناختیم و یاد گرفتیم که Tokenizer چگونه با الگوریتم‌های Subword متن را پردازش می‌کند. در پروژه‌ی عملی یک Fine-tuning کامل BERT روی دیتاست IMDB پیاده‌سازی کردیم که الگویی است که می‌توانید برای هر وظیفه‌ی NLP دیگری هم استفاده کنید.

---

# 🎯 تمرین‌های پایان فصل

۱. به‌جای BERT، از DistilBERT که نسخه‌ی فشرده‌شده‌ی آن است استفاده کنید و زمان آموزش و دقت را مقایسه کنید.

۲. مدل Fine-tune شده را با Pipeline روی جملات فارسی ترجمه‌شده به انگلیسی آزمایش کنید.

۳. از یک مدل BERT چندزبانه (مثل `bert-base-multilingual-cased`) برای متن فارسی استفاده کنید.

۴. خروجی لایه‌های مختلف BERT را استخراج کنید و با t-SNE در دو بُعد تجسم کنید.

۵. وظیفه‌ی Named Entity Recognition (شناسایی موجودیت‌های نامدار) را با استفاده از مدل‌های آماده‌ی Hugging Face پیاده‌سازی کنید.

---

در فصل ۲۴ به موضوع مهم **Fine-tuning پیشرفته‌ی مدل‌های زبانی بزرگ** می‌پردازیم و روش‌های کارآمد مثل LoRA و QLoRA را یاد می‌گیریم که آموزش مدل‌های بزرگ را حتی با منابع محدود ممکن می‌سازند.
