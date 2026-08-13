

# فصل ۲۶ 🎯 پروژه‌ی جامع NLP: دستیار اسنادی با RAG و Fine-tuning

## 🎯 اهداف فصل

این فصل پروژه‌ی جامع بخش چهارم کتاب است. مفاهیم فصل‌های ۱۹ تا ۲۵ یعنی ترنسفورمر، BERT، GPT، Fine-tuning پیشرفته، و RAG را در یک پروژه‌ی کامل و کاربردی ترکیب می‌کنیم: ساخت یک دستیار اسنادی هوشمند که می‌تواند بر اساس محتوای یک مجموعه‌ی اسناد واقعی به سؤالات کاربر پاسخ دهد. این دقیقاً همان نوع سامانه‌ای است که شرکت‌های بزرگ برای ساخت چت‌بات‌های داخلی، دستیارهای پشتیبانی مشتری، و ابزارهای جستجوی هوشمند در داده‌های اختصاصی خود استفاده می‌کنند.

---

# 🏗️ معماری کامل سامانه

سامانه‌ای که می‌سازیم از سه بخش اصلی تشکیل شده است:

1. **بخش اول: Indexing (نمایه‌سازی)**  
   اسناد را بارگذاری، به قطعات کوچک‌تر تقسیم، و بردار آن‌ها را در یک پایگاه‌داده‌ی برداری ذخیره می‌کند.

2. **بخش دوم: Retrieval (بازیابی)**  
   سؤال کاربر را بردارنگاری کرده، نزدیک‌ترین قطعات اسناد را پیدا می‌کند، و آن‌ها را به مدل زبانی می‌دهد.

3. **بخش سوم: Generation (تولید)**  
   مدل زبانی بر اساس سؤال و اسناد بازیابی‌شده، پاسخ نهایی را تولید می‌کند.

```python
import torch
import numpy as np
from typing import List, Dict, Tuple
import os
import json
import re

# ---- نصب کتابخانه‌های موردنیاز ----
# pip install transformers sentence-transformers faiss-cpu

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    from sentence_transformers import SentenceTransformer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("نصب: pip install transformers sentence-transformers")

# ---- بخش اول: آماده‌سازی اسناد ----

# مجموعه‌ی اسناد نمونه (در پروژه‌ی واقعی از PDF یا پایگاه داده بارگذاری کنید)
SAMPLE_DOCUMENTS = [
    {
        "id": "doc_1",
        "title": "مقدمه‌ای بر یادگیری عمیق",
        "content": """یادگیری عمیق شاخه‌ای از یادگیری ماشین است که از شبکه‌های عصبی
        با چندین لایه برای یادگیری از داده‌ها استفاده می‌کند. این تکنیک در
        بینایی ماشین، پردازش زبان طبیعی و بسیاری از حوزه‌های دیگر انقلابی
        ایجاد کرده است. شبکه‌های عصبی عمیق می‌توانند ویژگی‌های پیچیده را
        به‌صورت خودکار از داده‌های خام یاد بگیرند."""
    },
    {
        "id": "doc_2",
        "title": "شبکه‌های کانولوشنی",
        "content": """شبکه‌های عصبی کانولوشنی یا CNN برای پردازش داده‌های تصویری
        طراحی شده‌اند. این شبکه‌ها از لایه‌های کانولوشنی، Pooling و تمام‌متصل
        تشکیل شده‌اند. معماری‌های مشهور شامل ResNet، VGG و EfficientNet هستند.
        CNN ها در تشخیص تصویر، تشخیص اشیاء و تقسیم‌بندی تصویر کاربرد دارند."""
    },
    {
        "id": "doc_3",
        "title": "ترنسفورمر و مدل‌های زبانی",
        "content": """معماری ترنسفورمر در سال ۲۰۱۷ معرفی شد و انقلابی در NLP ایجاد کرد.
        مدل‌هایی مثل BERT و GPT بر پایه‌ی ترنسفورمر ساخته شده‌اند. BERT از
        معماری Encoder و GPT از معماری Decoder استفاده می‌کند. این مدل‌ها روی
        میلیاردها کلمه پیش‌آموزش دیده‌اند و می‌توانند برای وظایف مختلف Fine-tune شوند."""
    },
    {
        "id": "doc_4",
        "title": "بهینه‌سازی در یادگیری عمیق",
        "content": """بهینه‌سازهای مختلفی برای آموزش شبکه‌های عصبی وجود دارد.
        SGD ساده‌ترین روش است اما کند است. Adam که Momentum و RMSProp را ترکیب
        می‌کند، رایج‌ترین انتخاب در یادگیری عمیق است. نرخ یادگیری یکی از
        مهم‌ترین ابرپارامترها است و معمولاً از Scheduler برای تنظیم آن استفاده می‌شود."""
    },
    {
        "id": "doc_5",
        "title": "جلوگیری از بیش‌برازش",
        "content": """بیش‌برازش وقتی رخ می‌دهد که مدل روی داده‌ی آموزشی خوب عمل می‌کند
        اما روی داده‌ی جدید ضعیف است. روش‌های جلوگیری شامل Dropout، Batch
        Normalization، Early Stopping، L1/L2 Regularization و Data Augmentation
        هستند. نسبت داده‌ی آموزشی به اعتبارسنجی هم اهمیت دارد."""
    }
]


class DocumentProcessor:
    """پردازش و تقسیم اسناد به قطعات کوچک‌تر"""

    def __init__(self, chunk_size=200, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_into_chunks(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.overlap
        return chunks

    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        processed = []
        for doc in documents:
            chunks = self.split_into_chunks(doc['content'])
            for i, chunk in enumerate(chunks):
                processed.append({
                    'id': f"{doc['id']}_chunk_{i}",
                    'doc_id': doc['id'],
                    'title': doc['title'],
                    'content': chunk
                })
        return processed


class VectorStore:
    """پایگاه‌داده‌ی برداری ساده با جستجوی کسینوسی"""

    def __init__(self):
        self.vectors = []
        self.metadata = []

    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray):
        self.vectors = embeddings
        self.metadata = chunks

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict]:
        if len(self.vectors) == 0:
            return []

        # محاسبه‌ی شباهت کسینوسی
        norms_db = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms_db = np.where(norms_db == 0, 1, norms_db)
        normalized_db = self.vectors / norms_db

        norm_q = np.linalg.norm(query_vector)
        if norm_q == 0:
            norm_q = 1
        normalized_q = query_vector / norm_q

        similarities = normalized_db @ normalized_q
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                **self.metadata[idx],
                'similarity': float(similarities[idx])
            })
        return results


class SimpleEmbedder:
    """Embedder ساده بر اساس TF-IDF (جایگزین Sentence Transformer)"""

    def __init__(self, vocab_size=500):
        self.vocab = {}
        self.vocab_size = vocab_size

    def fit(self, texts: List[str]):
        word_counts = {}
        for text in texts:
            for word in text.lower().split():
                word = re.sub(r'[^\w]', '', word)
                if word:
                    word_counts[word] = word_counts.get(word, 0) + 1

        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        self.vocab = {w: i for i, (w, _) in enumerate(sorted_words[:self.vocab_size])}

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.vocab_size)
        words = text.lower().split()
        for word in words:
            word = re.sub(r'[^\w]', '', word)
            if word in self.vocab:
                vec[self.vocab[word]] += 1
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])


class RAGSystem:
    """سامانه‌ی کامل RAG"""

    def __init__(self):
        self.doc_processor = DocumentProcessor(chunk_size=100, overlap=20)
        self.vector_store = VectorStore()
        self.embedder = SimpleEmbedder(vocab_size=300)
        self.chunks = []

    def index_documents(self, documents: List[Dict]):
        print(f"نمایه‌سازی {len(documents)} سند...")
        self.chunks = self.doc_processor.process_documents(documents)
        print(f"تعداد قطعات ساخته‌شده: {len(self.chunks)}")

        texts = [c['content'] for c in self.chunks]
        self.embedder.fit(texts)
        embeddings = self.embedder.embed_batch(texts)
        self.vector_store.add_documents(self.chunks, embeddings)
        print("نمایه‌سازی کامل شد.\n")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        query_vec = self.embedder.embed(query)
        return self.vector_store.search(query_vec, top_k)

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        context = "\n\n".join([
            f"[منبع: {c['title']}]\n{c['content']}"
            for c in context_chunks
        ])
        prompt = f"""بر اساس اطلاعات زیر به سؤال پاسخ دهید:

{context}

سؤال: {query}

پاسخ: """
        # در پروژه‌ی واقعی از یک LLM برای تولید پاسخ استفاده کنید
        # اینجا برای سادگی پاسخ اول را برمی‌گردانیم
        if context_chunks:
            best = context_chunks[0]
            return (f"بر اساس سند «{best['title']}»: "
                    f"{best['content'][:200]}...")
        return "اطلاعات کافی برای پاسخ به این سؤال یافت نشد."

    def ask(self, query: str, top_k: int = 3) -> Dict:
        retrieved = self.retrieve(query, top_k)
        answer = self.generate_answer(query, retrieved)
        return {
            'query': query,
            'answer': answer,
            'sources': [
                {'title': r['title'], 'similarity': round(r['similarity'], 3)}
                for r in retrieved
            ]
        }


# ---- اجرای سامانه ----
print("=== ساخت سامانه‌ی RAG ===\n")

rag = RAGSystem()
rag.index_documents(SAMPLE_DOCUMENTS)

test_queries = [
    "شبکه‌های کانولوشنی چطور کار می‌کنند؟",
    "چطور از بیش‌برازش جلوگیری کنیم؟",
    "تفاوت BERT و GPT چیست؟",
    "بهترین بهینه‌ساز برای یادگیری عمیق کدام است؟"
]

print("=== نتایج پرسش‌وپاسخ ===\n")
for query in test_queries:
    result = rag.ask(query)
    print(f"سؤال: {result['query']}")
    print(f"پاسخ: {result['answer']}")
    print(f"منابع استفاده‌شده:")
    for src in result['sources']:
        print(f"  - {src['title']} (شباهت: {src['similarity']})")
    print("-" * 60)

print("\n=== ارزیابی سامانه ===\n")
eval_pairs = [
    ("CNN چیست؟", "doc_2"),
    ("ترنسفورمر چه سالی معرفی شد؟", "doc_3"),
    ("Dropout چیست؟", "doc_5"),
]

correct = 0
for question, expected_doc_id in eval_pairs:
    retrieved = rag.retrieve(question, top_k=1)
    if retrieved and expected_doc_id in retrieved[0]['doc_id']:
        correct += 1
        status = "✅"
    else:
        status = "❌"
    print(f"{status} سؤال: {question}")
    if retrieved:
        print(f"   بازیابی‌شده: {retrieved[0]['title']}")

print(f"\nدقت بازیابی: {correct}/{len(eval_pairs)} = "
      f"{100*correct/len(eval_pairs):.1f}%")
```

---

# ❓ سؤالات تستی

## سؤال ۱

در معماری RAG، چرا اسناد به قطعات کوچک‌تر (Chunk) تقسیم می‌شوند؟

الف) چون مدل‌های Embedding نمی‌توانند متن بلند را پردازش کنند  
ب) چون قطعات کوچک‌تر بردارهای دقیق‌تری تولید می‌کنند، جستجوی بازیابی دقیق‌تر است، و کنترل بهتری روی محتوای ارسالی به LLM وجود دارد  
ج) چون ذخیره‌سازی قطعات کوچک ارزان‌تر است  
د) چون LLM فقط می‌تواند متن کوتاه تولید کند  

## سؤال ۲

چه تفاوتی بین RAG و Fine-tuning برای اضافه کردن دانش جدید به یک LLM وجود دارد؟

الف) RAG و Fine-tuning کاملاً یکسان هستند  
ب) Fine-tuning دانش را در وزن‌های مدل می‌سوزاند و نیاز به بازآموزش دارد، در حالی که RAG دانش را خارج از مدل نگه می‌دارد و بدون آموزش مجدد می‌توان آن را به‌روز کرد  
ج) Fine-tuning سریع‌تر از RAG است  
د) RAG فقط برای متون انگلیسی کار می‌کند  

---

# ✅ پاسخ‌ها

## پاسخ سؤال ۱: گزینه ب

تقسیم به قطعات کوچک چند مزیت دارد: بردارهای قطعات کوچک دقیق‌تر معنای محلی را نمایش می‌دهند، می‌توان دقیقاً بخش مرتبط را پیدا کرد نه کل سند، و مدل LLM Context Window محدودی دارد که قطعات کوچک مدیریت آن را آسان‌تر می‌کند.

## پاسخ سؤال ۲: گزینه ب

Fine-tuning دانش را در پارامترهای مدل جاسازی می‌کند که برای تغییر آن باید مدل را دوباره آموزش داد. RAG دانش را در یک پایگاه‌داده‌ی خارجی نگه می‌دارد که به‌روزرسانی آن فقط نیاز به اضافه کردن اسناد جدید به پایگاه‌داده دارد.

---

# 📝 خلاصه فصل

در این فصل یک سامانه‌ی کامل RAG از صفر پیاده‌سازی کردیم. دیدیم که RAG با ترکیب بازیابی هوشمند اسناد و تولید متن با LLM، محدودیت دانش ثابت مدل‌های زبانی را برطرف می‌کند. سه مرحله‌ی اصلی (Indexing، Retrieval، Generation) را پیاده‌سازی کردیم و یک ارزیابی ساده روی دقت بازیابی انجام دادیم.

---

# 🎯 تمرین‌های پایان فصل

۱. پایگاه‌داده‌ی برداری را با FAISS جایگزین کنید که جستجوی بسیار سریع‌تری دارد.  
۲. Sentence Transformers را به‌جای Embedder ساده‌ی این فصل استفاده کنید.  
۳. اسناد واقعی PDF را با PyPDF2 بارگذاری کنید و به سامانه اضافه کنید.  
۴. یک رابط کاربری ساده با Gradio بسازید که کاربر بتواند سؤال بپرسد و پاسخ بگیرد.  
۵. دقت بازیابی را با معیار Recall@K ارزیابی کنید.

---
