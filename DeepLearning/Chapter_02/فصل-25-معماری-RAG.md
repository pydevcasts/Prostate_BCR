# فصل ۲۵ 📚 معماری بازیابی-افزوده یا RAG (Retrieval-Augmented Generation)

## 🎯 اهداف فصل

مدل‌های زبانی بزرگ که تاکنون بررسی کردیم، یک محدودیت اساسی دارند: دانش آن‌ها ایستا است و فقط به آنچه در طول پیش‌آموزش دیده‌اند محدود می‌شود. اگر از GPT-4 بخواهید آخرین اخبار را توضیح دهد یا از اسناد داخلی شرکت شما پاسخ بدهد، قادر نخواهد بود. معماری RAG که مخفف Retrieval-Augmented Generation است، این مشکل را با یک ایده‌ی ظریف حل می‌کند: قبل از تولید پاسخ، ابتدا اطلاعات مرتبط را از یک پایگاه دانش بازیابی کن و آن را به عنوان زمینه در اختیار مدل بگذار. در پایان این فصل خواهید توانست محدودیت دانش ایستای LLM را توضیح دهید، مفهوم Embedding متن و شباهت کسینوسی را درک کنید، ساختار Vector Database را بشناسید، یک سامانه‌ی RAG کامل از صفر تا انتها پیاده‌سازی کنید، و تفاوت‌های RAG ساده و RAG پیشرفته را بدانید.

---

# ⚠️ محدودیت‌های دانش ایستا در LLM

وقتی یک مدل زبانی مثل GPT یا LLaMA آموزش می‌بیند، دانش آن به آنچه در داده‌های پیش‌آموزش بوده محدود می‌شود. این محدودیت به چند شکل ظاهر می‌شود. نخست، مدل از رویدادهایی که بعد از تاریخ قطع داده‌هایش اتفاق افتاده بی‌اطلاع است. دوم، مدل به اسناد خصوصی و اختصاصی شما دسترسی ندارد؛ اگر بخواهید یک دستیار هوشمند بسازید که از دستنامه‌ی محصولات شرکتتان پاسخ بدهد، Fine-tuning ممکن است گران و کند باشد، و مدل همچنان ممکن است اطلاعات را «توهم» کند. سوم، حتی اگر Fine-tuning کنید، برای هر بار به‌روزرسانی اسناد باید دوباره Fine-tuning کنید که عملی نیست.

RAG هر سه این مشکل را با یک معماری جدا حل می‌کند: به‌جای اینکه دانش را درون مدل بدوزیم، آن را بیرون از مدل در یک پایگاه دانش نگه می‌داریم و در زمان نیاز بازیابی می‌کنیم.

---

# 🧭 معماری کلی RAG

یک سامانه‌ی RAG از دو فاز اصلی تشکیل می‌شود. فاز اول که فاز برون‌خط (Offline) نامیده می‌شود، قبل از استفاده‌ی واقعی اتفاق می‌افتد: اسناد منبع (مثل PDF، صفحات وب، یا اسناد متنی) به تکه‌های کوچک‌تر (Chunk) تقسیم می‌شوند، هر تکه به یک بردار عددی (Embedding) تبدیل می‌شود که معنای آن را نمایش می‌دهد، و این بردارها در یک پایگاه‌داده‌ی برداری (Vector Database) ذخیره می‌شوند.

فاز دوم که فاز برخط (Online) است، برای هر سؤال کاربر اتفاق می‌افتد: سؤال کاربر هم به بردار تبدیل می‌شود، شباهت این بردار با تمام بردارهای موجود در پایگاه‌داده محاسبه می‌شود و مرتبط‌ترین تکه‌ها بازیابی می‌شوند، سپس این تکه‌ها به همراه سؤال اصلی به LLM داده می‌شوند تا پاسخ نهایی را تولید کند.

📷 [تصویر اینجا قرار گیرد: نمودار کامل معماری RAG با دو فاز: فاز آفلاین (تبدیل اسناد به Embedding و ذخیره در Vector DB) و فاز آنلاین (تبدیل سؤال به Embedding، بازیابی اسناد مشابه، و ارسال به LLM)]

---

# 🔢 بردار متن (Text Embedding) و شباهت کسینوسی

Embedding متن فرآیندی است که در آن یک جمله یا پاراگراف به یک بردار عددی چندبعدی تبدیل می‌شود که معنای آن را در فضای برداری نمایش می‌دهد. جملاتی که معنای مشابهی دارند، بردارهای نزدیکی در این فضا خواهند داشت. مدل‌هایی مثل sentence-transformers به‌خصوص برای این وظیفه طراحی شده‌اند.

برای اندازه‌گیری شباهت دو بردار، معمولاً از شباهت کسینوسی (Cosine Similarity) استفاده می‌شود که زاویه‌ی میان دو بردار را اندازه می‌گیرد:

```
cosine_similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

در این فرمول، A · B حاصل‌ضرب داخلی دو بردار است و ‖A‖ و ‖B‖ طول آن‌ها. مقدار این معیار بین منفی یک و مثبت یک است. مقدار نزدیک به یک یعنی دو جمله معنای بسیار مشابهی دارند، مقدار نزدیک به صفر یعنی بی‌ارتباطند، و مقدار منفی یعنی معنای متضاد دارند.

یک مثال ساده: بردار جمله‌ی «یادگیری عمیق چیست؟» باید به بردار «یادگیری عمیق شاخه‌ای از هوش مصنوعی است» نزدیک باشد، اما از بردار «قیمت طلا امروز چقدر است؟» دور باشد.

---

# 🗄️ پایگاه‌داده‌ی برداری (Vector Database)

پایگاه‌داده‌های برداری برای ذخیره و جستجوی سریع میلیون‌ها بردار طراحی شده‌اند. برخلاف پایگاه‌داده‌های معمولی که بر اساس مقادیر دقیق جستجو می‌کنند، پایگاه‌داده‌های برداری جستجوی نزدیک‌ترین همسایه‌ی تقریبی (Approximate Nearest Neighbor) را بهینه کرده‌اند. رایج‌ترین گزینه‌های موجود شامل FAISS که توسط متا توسعه یافته و رایگان است، Chroma که ساده‌ترین گزینه برای شروع است، Pinecone که یک سرویس ابری مدیریت‌شده است، و Weaviate یا Qdrant که گزینه‌های متن‌باز با قابلیت‌های بیشتر هستند می‌شوند. در پروژه‌ی این فصل از FAISS و Chroma استفاده می‌کنیم.

---

# 💻 پروژه‌ی عملی فصل: ساخت یک سامانه‌ی پرسش‌وپاسخ مبتنی بر RAG

در این پروژه یک سامانه‌ی RAG کامل می‌سازیم که می‌تواند از مجموعه‌ای از اسناد متنی (مثل مقالات ویکی‌پدیا یا اسناد PDF) پرسش‌وپاسخ کند.

```python
# pip install sentence-transformers faiss-cpu transformers langchain chromadb

import numpy as np
import torch
from typing import List, Tuple

# ---- بخش اول: پیاده‌سازی دستی برای درک مفاهیم ----

class SimpleVectorStore:
    """
    یک پایگاه‌داده‌ی برداری ساده برای نمایش مفاهیم RAG.
    در پروژه‌های واقعی از FAISS یا Chroma استفاده کنید.
    """

    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []

    def add_documents(self, docs: List[str], embedder, meta: List[dict] = None):
        """اضافه کردن اسناد جدید به پایگاه‌داده"""
        new_embeddings = embedder(docs)
        self.documents.extend(docs)
        self.embeddings.extend(new_embeddings)
        if meta:
            self.metadata.extend(meta)
        else:
            self.metadata.extend([{}] * len(docs))
        print(f"  {len(docs)} سند اضافه شد. مجموع: {len(self.documents)}")

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """محاسبه‌ی شباهت کسینوسی بین دو بردار"""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple]:
        """جستجوی مرتبط‌ترین اسناد"""
        if not self.embeddings:
            return []

        similarities = [
            self.cosine_similarity(query_embedding, emb)
            for emb in self.embeddings
        ]

        # مرتب‌سازی بر اساس شباهت به‌صورت نزولی
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((
                self.documents[idx],
                similarities[idx],
                self.metadata[idx]
            ))
        return results


# ---- بخش دوم: پیاده‌سازی RAG با کتابخانه‌های واقعی ----

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# ---- اسناد نمونه ----
sample_documents = [
    {
        "text": (
            "Deep learning is a subset of machine learning that uses "
            "neural networks with multiple layers. These networks can "
            "learn complex patterns from large datasets automatically "
            "without explicit feature engineering."
        ),
        "source": "deep_learning_intro.txt"
    },
    {
        "text": (
            "LSTM (Long Short-Term Memory) networks are a type of "
            "recurrent neural network designed to learn long-term "
            "dependencies. They use gates to control information flow: "
            "forget gate, input gate, and output gate."
        ),
        "source": "lstm_explanation.txt"
    },
    {
        "text": (
            "The Transformer architecture was introduced in 2017 in the "
            "paper Attention Is All You Need. It uses self-attention "
            "mechanisms instead of recurrence, allowing parallel "
            "processing of sequences."
        ),
        "source": "transformer_paper.txt"
    },
    {
        "text": (
            "BERT (Bidirectional Encoder Representations from Transformers) "
            "is pre-trained on masked language modeling. It reads text "
            "bidirectionally, making it excellent for understanding context."
        ),
        "source": "bert_overview.txt"
    },
    {
        "text": (
            "Convolutional Neural Networks (CNNs) are specialized for "
            "processing grid-like data such as images. They use convolutional "
            "layers to automatically detect spatial hierarchies of features."
        ),
        "source": "cnn_guide.txt"
    },
    {
        "text": (
            "The backpropagation algorithm is used to train neural networks. "
            "It computes gradients of the loss function with respect to each "
            "weight using the chain rule of calculus."
        ),
        "source": "backprop_tutorial.txt"
    },
    {
        "text": (
            "Gradient descent is an optimization algorithm that minimizes the "
            "loss function by iteratively moving in the direction of steepest "
            "descent. Adam optimizer combines momentum and adaptive learning rates."
        ),
        "source": "optimization_methods.txt"
    },
    {
        "text": (
            "Overfitting occurs when a model learns training data too well, "
            "including noise. Techniques to prevent overfitting include "
            "dropout, L2 regularization, early stopping, and data augmentation."
        ),
        "source": "regularization_guide.txt"
    }
]


class RAGSystem:
    """سامانه‌ی کامل RAG با قابلیت تنظیم مدل Embedding و مدل زبانی"""

    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        self.vector_store = SimpleVectorStore()
        self.embedding_model = None
        self.embedding_dim = 0

        if ST_AVAILABLE:
            print(f"بارگذاری مدل Embedding: {embedding_model_name}")
            self.embedding_model = SentenceTransformer(embedding_model_name)
            # تعیین بُعد بردار با یک نمونه‌ی آزمایشی
            test_emb = self.embedding_model.encode(["test"])
            self.embedding_dim = test_emb.shape[1]
            print(f"بُعد بردار Embedding: {self.embedding_dim}")
        else:
            print("SentenceTransformer نصب نیست. از Embedding ساختگی استفاده می‌شود.")
            self.embedding_dim = 64

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """تبدیل متون به بردار"""
        if self.embedding_model:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            return [emb for emb in embeddings]
        else:
            # Embedding ساختگی برای نمایش ساختار
            np.random.seed(42)
            return [np.random.randn(self.embedding_dim) for _ in texts]

    def chunk_text(self, text: str, chunk_size: int = 200,
                   overlap: int = 50) -> List[str]:
        """
        تقسیم متن به تکه‌های کوچک‌تر با overlap برای حفظ زمینه.
        overlap یعنی هر تکه‌ی جدید چند کلمه از آخر تکه‌ی قبلی را دارد.
        """
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start += chunk_size - overlap
        return chunks

    def index_documents(self, documents: List[dict]):
        """ایندکس‌گذاری اسناد در پایگاه‌داده‌ی برداری"""
        print("\nایندکس‌گذاری اسناد...")
        all_chunks = []
        all_metadata = []

        for doc in documents:
            chunks = self.chunk_text(doc["text"])
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": doc["source"],
                    "chunk_idx": i,
                    "total_chunks": len(chunks)
                })

        self.vector_store.add_documents(
            all_chunks, self.embed_texts, all_metadata
        )

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple]:
        """بازیابی مرتبط‌ترین اسناد برای یک سؤال"""
        query_embedding = self.embed_texts([query])[0]
        results = self.vector_store.search(query_embedding, top_k)
        return results

    def generate_prompt(self, query: str, context_chunks: List[Tuple]) -> str:
        """ساخت Prompt برای LLM با اطلاعات بازیابی‌شده"""
        context_text = "\n\n".join([
            f"[منبع: {meta.get('source', 'ناشناخته')}]\n{chunk}"
            for chunk, score, meta in context_chunks
        ])

        prompt = f"""با استفاده از اطلاعات زیر به سؤال پاسخ بده.
فقط از اطلاعات ارائه‌شده استفاده کن. اگر پاسخ در اطلاعات نبود، بگو «اطلاعات کافی ندارم».

اطلاعات زمینه:
{context_text}

سؤال: {query}

پاسخ:"""
        return prompt

    def answer(self, query: str, top_k: int = 3,
               llm_fn=None) -> dict:
        """پاسخ به سؤال با استفاده از RAG"""

        # بازیابی اسناد مرتبط
        retrieved = self.retrieve(query, top_k)

        # ساخت Prompt
        prompt = self.generate_prompt(query, retrieved)

        # تولید پاسخ (یا نمایش Prompt اگر LLM نداشتیم)
        if llm_fn:
            answer_text = llm_fn(prompt)
        else:
            answer_text = f"[پرامپت آماده برای LLM]\n{prompt[:300]}..."

        return {
            "query": query,
            "answer": answer_text,
            "sources": [meta.get('source') for _, _, meta in retrieved],
            "retrieved_chunks": retrieved
        }


# ---- اجرای سامانه‌ی RAG ----
print("=== ساخت سامانه‌ی RAG ===\n")

rag = RAGSystem()
rag.index_documents(sample_documents)

# سؤالات آزمایشی
test_queries = [
    "How does LSTM handle long-term dependencies?",
    "What is the difference between BERT and GPT?",
    "How does backpropagation work?"
]

print("\n=== نتایج جستجوی مرتبط‌ترین اسناد ===\n")

for query in test_queries:
    print(f"سؤال: {query}")
    retrieved = rag.retrieve(query, top_k=2)
    for i, (chunk, score, meta) in enumerate(retrieved):
        print(f"  نتیجه {i+1} | شباهت: {score:.3f} | منبع: {meta.get('source')}")
        print(f"  متن: {chunk[:100]}...")
    print()


# ---- بخش سوم: RAG با FAISS برای مقیاس بالا ----

if FAISS_AVAILABLE and ST_AVAILABLE:
    print("\n=== RAG با FAISS (نسخه‌ی مقیاس‌پذیر) ===\n")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [doc["text"] for doc in sample_documents]
    embeddings = embedding_model.encode(texts)
    embeddings_np = np.array(embeddings).astype('float32')
    dimension = embeddings_np.shape[1]

    # ایجاد ایندکس FAISS
    # IndexFlatIP برای شباهت کسینوسی (پس از نرمال‌سازی)
    faiss.normalize_L2(embeddings_np)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)
    print(f"FAISS ایندکس ساخته شد با {index.ntotal} سند")

    # جستجو با FAISS
    test_query = "How do transformers work?"
    query_emb = embedding_model.encode([test_query]).astype('float32')
    faiss.normalize_L2(query_emb)

    scores, indices = index.search(query_emb, k=3)
    print(f"\nجستجوی FAISS برای: '{test_query}'")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        print(f"  رتبه {rank+1} | شباهت: {score:.3f}")
        print(f"  متن: {texts[idx][:100]}...")

else:
    print("\nبرای اجرای RAG با FAISS:")
    print("pip install faiss-cpu sentence-transformers")


# ---- بخش چهارم: ارزیابی کیفیت بازیابی ----

def evaluate_retrieval_quality(rag_system, test_cases):
    """
    ارزیابی کیفیت بازیابی با مقایسه‌ی منبع بازیابی‌شده
    با منبع مورد انتظار
    """
    print("\n=== ارزیابی کیفیت بازیابی ===\n")
    correct = 0

    for query, expected_source in test_cases:
        retrieved = rag_system.retrieve(query, top_k=1)
        if retrieved:
            actual_source = retrieved[0][2].get('source', '')
            is_correct = expected_source in actual_source
            status = "✓" if is_correct else "✗"
            print(f"{status} سؤال: {query[:50]}...")
            print(f"  انتظار: {expected_source} | بازیابی: {actual_source}")
            if is_correct:
                correct += 1

    accuracy = 100 * correct / len(test_cases)
    print(f"\nدقت بازیابی: {accuracy:.1f}%")
    return accuracy


test_cases = [
    ("What are LSTM gates?", "lstm_explanation.txt"),
    ("How does backpropagation compute gradients?", "backprop_tutorial.txt"),
    ("What is overfitting?", "regularization_guide.txt"),
    ("How do CNNs detect features in images?", "cnn_guide.txt"),
]

evaluate_retrieval_quality(rag, test_cases)
```

---

# ❓ سؤالات تستی

### سؤال ۱

چرا در سامانه‌ی RAG، هنگام تقسیم اسناد به Chunk‌ها از Overlap استفاده می‌شود؟

الف) برای کاهش تعداد Chunk‌ها و صرفه‌جویی در حافظه

ب) برای اطمینان از اینکه اطلاعات مهمی که در مرز دو تکه قرار دارند از بین نروند، زیرا بدون Overlap ممکن است یک جمله‌ی مهم در انتهای یک تکه و ابتدای تکه‌ی بعدی قطع شود

ج) چون Vector Database به Chunk‌های با طول یکسان نیاز دارد

د) برای بهبود عملکرد محاسباتی

### سؤال ۲

چرا RAG می‌تواند جایگزین بهتری نسبت به Fine-tuning برای افزودن دانش جدید به یک LLM باشد؟

الف) چون RAG هیچ‌وقت اشتباه نمی‌کند

ب) چون RAG می‌تواند هر زمان با به‌روزرسانی اسناد در پایگاه‌داده دانش مدل را بدون آموزش مجدد به‌روز کند، منابع اطلاعات را قابل‌ردیابی می‌کند، و هزینه‌ی محاسباتی بسیار کمتری نسبت به Fine-tuning دارد

ج) چون RAG از LLM نیازی ندارد

د) چون Fine-tuning اسناد جدید را به‌طور خودکار یاد می‌گیرد

---

## ✅ پاسخ‌ها

**پاسخ سؤال ۱: گزینه ب**

بدون Overlap، اگر یک جمله‌ی کلیدی دقیقاً در مرز دو Chunk قرار گیرد، هیچ‌کدام از آن دو Chunk اطلاعات کافی درباره‌ی آن جمله نخواهند داشت. Overlap با تکرار بخشی از متن در هر دو Chunk مجاور، این مشکل را حل می‌کند و تضمین می‌کند که زمینه‌ی لازم برای درک هر اطلاعات مهم در حداقل یک Chunk کامل موجود باشد.

**پاسخ سؤال ۲: گزینه ب**

Fine-tuning دانش را درون وزن‌های مدل می‌دوزد و برای به‌روزرسانی نیاز به آموزش مجدد دارد. RAG پایگاه دانش را جدا از مدل نگه می‌دارد؛ برای افزودن اطلاعات جدید فقط باید اسناد جدید را Embed کرده و به Vector Database اضافه کرد. علاوه بر این، RAG قادر است منبع هر اطلاعات را نشان دهد که برای اعتبارسنجی پاسخ‌ها بسیار مهم است.

---

# 📝 خلاصه فصل

در این فصل معماری RAG را به‌عنوان راه‌حلی برای محدودیت دانش ایستای LLM‌ها بررسی کردیم. دیدیم که RAG با تفکیک دانش از مدل و بازیابی پویا در زمان نیاز، چالش‌هایی مثل اطلاعات جدید، اسناد اختصاصی و به‌روزرسانی آسان را حل می‌کند. مفاهیم Embedding متن، شباهت کسینوسی، Vector Database و Chunking با Overlap را با فرمول و کد درک کردیم. یک سامانه‌ی RAG کامل از صفر پیاده‌سازی کردیم که شامل ایندکس‌گذاری اسناد، بازیابی مرتبط‌ترین تکه‌ها، ساخت Prompt غنی‌شده و ارزیابی کیفیت بازیابی می‌شود.

---

# 🎯 تمرین‌های پایان فصل

۱. سامانه‌ی RAG را با یک مجموعه از فایل‌های PDF واقعی (مثل مقالات یا کتاب‌های آموزشی) آزمایش کنید.

۲. اندازه‌ی Chunk و مقدار Overlap را تغییر دهید و تأثیر آن بر دقت بازیابی را بسنجید.

۳. از کتابخانه‌ی LangChain یا LlamaIndex برای ساخت همین سامانه استفاده کنید و کد را با نسخه‌ی دستی مقایسه کنید.

۴. یک مکانیزم Re-ranking اضافه کنید که نتایج بازیابی را با یک مدل Cross-Encoder دوباره رتبه‌بندی کند.

۵. سامانه‌ی RAG را با LLM واقعی (مثل یک مدل از Hugging Face یا API OpenAI) ترکیب کنید و یک چت‌بات مستندات بسازید.

---

در فصل ۲۶ به **پروژه‌ی جامع NLP** می‌رسیم؛ جایی که ترکیب Fine-tuning و RAG را در یک سامانه‌ی کاربردی کامل برای پرسش‌وپاسخ از اسناد فارسی پیاده‌سازی می‌کنیم.
