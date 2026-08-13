import os
import subprocess

os.chdir(r"D:/ML_Doc/DeepLearning/Chapter_02")

# دیکشنری نام‌های فارسی به انگلیسی
rename_map = {
    "فصل ۲۷  Autoencoder و Variational Autoencoder (VAE).md": "chapter-27-autoencoder-and-vae.md",
    "فصل_26دستیار اسنادی با RAG و Fine-tuning.md": "chapter-26-document-assistant-with-rag-and-fine-tuning.md",
    "فصل-01-آماده_سازی-محیط-و-ابزارهای-برنامه_نویسی.md": "chapter-01-environment-setup-and-programming-tools.md",
    "فصل-02-نورون-مصنوعی-و-شبکه_های-عصبی-پایه.md": "chapter-02-artificial-neurons-and-basic-neural-networks.md",
    "فصل-03-فرآیند-یادگیری-تابع-هزینه-و-گرادیان-نزولی.md": "chapter-03-learning-process-cost-function-and-gradient-descent.md",
    "فصل-04-الگوریتم-پس_انتشار-خطا.md": "chapter-04-backpropagation-algorithm.md",
    "فصل-05-بهینه_سازهای-پیشرفته.md": "chapter-05-advanced-optimizers.md",
    "فصل-06-منظم_سازی-و-جلوگیری-از-بیش_برازش.md": "chapter-06-regularization-and-overfitting-prevention.md",
    "فصل-07-ارزیابی-مدل-و-آماده_سازی-داده.md": "chapter-07-model-evaluation-and-data-preparation.md",
    "فصل-08-پروژه_ی-جامع-بخش-اول.md": "chapter-08-comprehensive-project-part-one.md",
    "فصل-09-مقدمه_ای-بر-پردازش-تصویر.md": "chapter-09-introduction-to-image-processing.md",
    "فصل-10-عملیات-کانولوشن-و-ساختار-CNN.md": "chapter-10-convolution-operation-and-cnn-architecture.md",
    "فصل-11-معماری_های-کلاسیک-CNN.md": "chapter-11-classic-cnn-architectures.md",
    "فصل-12-معماری_های-مدرن-CNN.md": "chapter-12-modern-cnn-architectures.md",
    "فصل-13-پروژه_ی-جامع-بینایی-ماشین.md": "chapter-13-computer-vision-comprehensive-project.md",
    "فصل-14-مقدمه_ی-داده_های-ترتیبی-و-RNN.md": "chapter-14-introduction-to-sequential-data-and-rnn.md",
    "فصل-15-مشکل-گرادیان-محوشونده-و-انفجاری.md": "chapter-15-vanishing-and-exploding-gradient-problem.md",
    "فصل-16-شبکه_ی-LSTM.md": "chapter-16-lstm-networks.md",
    "فصل-17-شبکه_ی-GRU-و-مقایسه-با-LSTM.md": "chapter-17-gru-networks-and-comparison-with-lstm.md",
    "فصل-18-پروژه_ی-جامع-بخش-سوم.md": "chapter-18-comprehensive-project-part-three.md",
    "فصل-19-مکانیزم-توجه-Attention.md": "chapter-19-attention-mechanism.md",
    "فصل-20-معماری-ترنسفورمر.md": "chapter-20-transformer-architecture.md",
    "فصل-21-پیش_پردازش-متن-و-بازنمایی-کلمات.md": "chapter-21-text-preprocessing-and-word-representation.md",
    "فصل-22-پردازش-زبان-طبیعی-با-RNN-LSTM.md": "chapter-22-nlp-with-rnn-and-lstm.md",
    "فصل-23-مدل_های-زبانی-BERT-و-GPT.md": "chapter-23-bert-and-gpt-language-models.md",
    "فصل-24-Fine-tuning-پیشرفته-LoRA-QLoRA.md": "chapter-24-advanced-fine-tuning-lora-and-qlora.md",
    "فصل-25-معماری-RAG.md": "chapter-25-rag-architecture.md",
    "فصل-28-شبکه_های-مولد-تخاصمی-GAN.md": "chapter-28-generative-adversarial-networks-gan.md",
    "فصل-29-یادگیری-انتقالی-پیشرفته-بینایی-ماشین (1).md": "chapter-29-advanced-transfer-learning-computer-vision.md",
    "فصل-30-استقرار-مدل-و-عملیاتی_سازی.md": "chapter-30-model-deployment-and-operationalization.md",
    "فصل-31-جمع_بندی-مسیر-یادگیری-و-بازار-کار.md": "chapter-31-learning-path-summary-and-job-market.md"
}

# تغییر نام با git mv
success_count = 0
for old_name, new_name in rename_map.items():
    try:
        # پیدا کردن فایل با نام دقیق
        if os.path.exists(old_name):
            subprocess.run(["git", "mv", old_name, new_name], check=True)
            print(f"✅ {old_name} → {new_name}")
            success_count += 1
        else:
            # جستجوی فایل با نام مشابه (حذف فاصله‌های اضافی)
            found = False
            for f in os.listdir("."):
                if f.strip() == old_name.strip():
                    subprocess.run(["git", "mv", f, new_name], check=True)
                    print(f"✅ {f} → {new_name}")
                    success_count += 1
                    found = True
                    break
            if not found:
                print(f"⚠️ فایل '{old_name}' پیدا نشد!")
    except Exception as e:
        print(f"❌ خطا در تغییر نام {old_name}: {e}")

print(f"\n📊 {success_count} از {len(rename_map)} فایل با موفقیت تغییر نام دادند.")