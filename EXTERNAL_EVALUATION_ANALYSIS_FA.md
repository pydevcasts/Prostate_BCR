# گزارش تحلیل افت عملکرد مدل در ارزیابی خارجی (GSE70769)

## خلاصه اجرایی

بررسی جامع نوت‌بوک `08_External _Evaluation.ipynb` و داده‌های مرتبط نشان می‌دهد که **AUC مدل در ارزیابی خارجی به 0.50 رسیده است** (معادل پیش‌بینی تصادفی)، در حالی که AUC داخلی حدود 0.82-0.87 است. این گزارش دلایل اصلی و راهکارهای اصلاحی را ارائه می‌دهد.

---

## ۱. بررسی کد و پایش داده‌ها (Data and Code Review)

### ۱.۱ عدم تطابق ویژگی‌ها (Feature Mismatch) - **مشکل اصلی**

**یافته کلیدی:** از 37 ویژگی انتخاب‌شده توسط PSO، تنها 32 ویژگی در داده GSE70769 موجود است:

```
Available: 32/37
Missing (5): ['Primary Lymph Node Presentation Assessment Ind-3_YES', 'GSAP', 
              'Patient Death Reason_Other, non-malignant disease', 'LINC00926', 'Margin_x_LymphNode']
```

**علت:**
- داده‌های TCGA از RNA-Seq (توالی‌یابی) استفاده می‌کنند
- داده‌های GSE70769 از Microarray (GPL10558 Illumina) استفاده می‌کند
- برخی ژن‌ها/ویژگی‌ها در پلتفرم Microarray وجود ندارند

**کد مشکل‌دار (Cell 4-5):**
```python
available = [f for f in selected_features if f in X_ext_eng.columns]
missing   = [f for f in selected_features if f not in X_ext_eng.columns]
# نتیجه: 5 ویژگی گم شده!
```

### ۱.۲ عدم نرمالیزاسیون Cross-Platform

**مشکل:** داده‌های TCGA و GSE70769 توزیع‌های کاملاً متفاوتی دارند:
- TCGA: داده‌های RNA-Seq با توزیع Count-based
- GSE70769: داده‌های Microarray با توزیع Intensity-based

**شواهد:**
```python
# هیچ نرمالیزاسیون cross-platform اعمال نشده است
X_eval = X_ext_eng[common_features]  # فقط زیرمجموعه‌گیری ساده
y_prob_ext = model.predict_proba(X_eval)[:, 1]  # پیش‌بینی مستقیم → خطا!
```

### ۱.۳ ویژگی‌های بالینی غیرقابل انتقال

**مشکل:** ویژگی‌های مهندسی‌شده بالینی در GSE70769 قابل محاسبه نیستند:
- `Gleason_Total` - نیاز به داده‌های پاتولوژی دارد
- `PSA_Pathway_Score` - نیاز به داده‌های PSA دارد
- `Margin_x_LymphNode` - نیاز به داده‌های جراحی دارد

در حال حاضر کد سعی می‌کند این ویژگی‌ها را از metadata استخراج کند (Cell 3)، اما:
```python
# خطوط 199-235: استخراج ویژگی‌های بالینی
gl_col = next((c for c in clin.columns if 'gleason' in c), None)
if gl_col:
    gl = clin[gl_col].apply(lambda x: int(''.join(filter(str.isdigit, str(x)))))
else:
    # مقدار پیش‌فرض 0 - باعث از دست رفتن اطلاعات می‌شود!
```

### ۱.۴ وضعیت فایل‌های داده

| فایل | وضعیت | تعداد نمونه | تعداد ویژگی |
|------|-------|------------|-------------|
| `GSE70769_family.soft.gz` | ✅ سالم (58MB) | 94 | ~31,426 پروب |
| `X_GSE70769.csv` | ✅ پردازش‌شده | 94 | 29,720 ژن |
| `y_GSE70769.csv` | ✅ پردازش‌شده | 94 | 1 هدف |
| توزیع BCR | ⚠️ نامتعادل | 45 مثبت / 49 منفی | - |

---

## ۲. تحلیل آماری و مفهومی

### ۲.۱ تفاوت‌های جمعیت‌شناختی

| ویژگی | TCGA-PRAD | GSE70769 | تأثیر |
|--------|-----------|----------|-------|
| **پلتفرم** | Illumina HiSeq (RNA-Seq) | Illumina HumanHT-12 (Microarray) | بالا |
| **تعداد نمونه** | ~500 بیمار | 94 بیمار | متوسط |
| **نوع نمونه** | Radical Prostatectomy | TURP/Prostatectomy | متوسط |
| **جمعیت** | آمریکای شمالی | اروپایی (انگلستان) | متوسط |
| **دوره زمانی** | 2000-2010 | 2005-2012 | کم |

### ۲.۲ Batch Effect شدید

**تحلیل:** تفاوت بین RNA-Seq و Microarray منجر به:
1. **توزیع متفاوت بیان ژن:** مقادیر خام قابل مقایسه نیستند
2. **Dynamic Range متفاوت:** RNA-Seq محدوده دینامیکی وسیع‌تری دارد
3. **Background Noise:** Microarray نویز زمینه بالاتری دارد

### ۲.۳ Overfitting احتمالی

**نشانه‌ها:**
- Internal CV AUC: 0.82-0.87 (بسیار خوب)
- External AUC: 0.50 (تصادفی!)
- تفاوت 0.32-0.37 نشان‌دهنده overfitting شدید است

**علل:**
1. مدل روی batch effect داده‌های TCGA overfit شده
2. ویژگی‌های بالینی خاص TCGA یاد گرفته شده
3. نرمالیزاسیون ناکافی در مرحله preprocessing

---

## ۳. خطاها و هشدارهای اجرا

### ۳.۱ خطاهای شناسایی‌شده

| خط/Cell | نوع | توضیح |
|---------|-----|-------|
| Cell 1 | ⚠️ Warning | نصب GEOparse در هر اجرا |
| Cell 5 | ❌ Critical | AUC = 0.50 (مدل تصادفی) |
| Cell 5 | ⚠️ Warning | `use_label_encoder=False` deprecated |
| Cell 5 | ⚠️ Warning | Refit on intersection only (32 features) |

### ۳.۲ سازگاری کتابخانه‌ها

```bash
# کتابخانه‌های کلیدی بررسی‌شده:
- pandas: ✅ سازگار
- numpy: ✅ سازگار  
- scikit-learn: ✅ سازگار
- xgboost: ⚠️ هشدار deprecation برای use_label_encoder
- GEOparse: ✅ کار می‌کند (اما فایل فشرده باید سالم باشد)
```

### ۳.۳ مشکل فایل فشرده (از لاگ کاربر)

```
EOFError: Compressed file ended before the end-of-stream marker was reached
```

**راه‌حل:** فایل باید دوباره دانلود شود:
```python
# حذف فایل خراب و دانلود مجدد
import requests
from pathlib import Path

soft_path = Path("core/data/external/GSE70769_family.soft.gz")
if soft_path.exists():
    soft_path.unlink()  # حذف فایل خراب

url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70769/soft/GSE70769_family.soft.gz"
response = requests.get(url, stream=True, timeout=180)
with open(soft_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=1024*1024):
        f.write(chunk)
```

---

## ۴. راهکارهای پیشنهادی (به ترتیب اولویت)

### 🔴 اولویت ۱: اعمال Cross-Platform Normalization

**روش پیشنهادی: Quantile Normalization یا Rank-based**

```python
# افزودن به Notebook 08 یا استفاده از Notebook 09
from src.validation import normalize_cross_platform, prepare_external_validation

# روش ۱: Quantile Normalization (توصیه‌شده)
X_train_norm, X_ext_norm = normalize_cross_platform(
    X_train_subset, X_ext_subset, method="quantile"
)

# روش ۲: Rank-based (مقاوم‌ترین روش)
X_train_norm, X_ext_norm = normalize_cross_platform(
    X_train_subset, X_ext_subset, method="rank"
)

# روش ۳: Z-score با آمار مرجع
X_train_norm, X_ext_norm = normalize_cross_platform(
    X_train_subset, X_ext_subset, method="zscore"
)
```

**انتظار بهبود:** +5-10% AUC

### 🟠 اولویت ۲: حذف ویژگی‌های بالینی غیرقابل انتقال

```python
# فقط از امضای ژنی استفاده کن (بدون ویژگی‌های بالینی)
gene_only_features = [f for f in selected_features 
                      if f not in clinical_feature_list]

X_train_gene = X_train[gene_only_features]
X_ext_gene = X_ext[gene_only_features]

# سپس نرمالیزاسیون cross-platform اعمال کن
```

**انتظار بهبود:** +2-5% AUC

### 🟡 اولویت ۳: Retraining روی ویژگی‌های مشترک

```python
# به جای استفاده از مدل frozen، روی ویژگی‌های مشترک retrain کن
from xgboost import XGBClassifier

common_features = list(set(X_train.columns) & set(X_ext.columns))
common_features = [f for f in common_features if not f.startswith(('Gleason', 'PSA', 'Margin'))]

# Retraining
clf = XGBClassifier(**best_params, random_state=42)
clf.fit(X_train[common_features], y_train)

# Evaluation
y_prob_ext = clf.predict_proba(X_ext[common_features])[:, 1]
auc_ext = roc_auc_score(y_ext, y_prob_ext)
```

**انتظار بهبود:** +5-10% AUC

### 🟢 اولویت ۴: استفاده از Notebook 09 (از قبل پیاده‌سازی شده)

```python
# Notebook 09_Improved_External_Validation.ipynb شامل تمام بهبودهاست
from src.validation import prepare_external_validation

X_train_aligned, X_ext_aligned, y_ext, used_features = prepare_external_validation(
    X_train, X_GSE70769, y_GSE70769,
    selected_features,
    exclude_clinical=True,           # حذف ویژگی‌های بالینی
    normalization_method="quantile"  # نرمالیزاسیون quantile
)

# حالا پیش‌بینی انجام بده
y_prob = model.predict_proba(X_ext_aligned[used_features])[:, 1]
```

---

## ۵. خروجی نهایی و مراحل بعدی

### ۵.۱ خلاصه دلایل افت عملکرد

| دلیل | شدت تأثیر | وضعیت فعلی |
|------|-----------|------------|
| Batch Effect (RNA-Seq vs Microarray) | 🔴 بسیار بالا | ❌ حل نشده |
| ویژگی‌های گم‌شده (5 از 37) | 🟠 بالا | ⚠️ Partial fix |
| ویژگی‌های بالینی غیرقابل انتقال | 🟠 بالا | ❌ حل نشده |
| عدم نرمالیزاسیون cross-platform | 🔴 بسیار بالا | ❌ حل نشده |
| Overfitting روی TCGA | 🟡 متوسط | ❌ حل نشده |

### ۵.۲ مراحل اصلاحی (Action Plan)

**مرحله ۱: آماده‌سازی داده‌ها**
```bash
# ۱.۱ اطمینان از سلامت فایل GSE70769
gunzip -t core/data/external/GSE70769_family.soft.gz

# ۱.۲ اگر خطا داد، دوباره دانلود کن
rm core/data/external/GSE70769_family.soft.gz
# اجرای مجدد Cell 3 از Notebook 08
```

**مرحله ۲: اجرای Notebook 09**
```python
# Notebook 09_Improved_External_Validation.ipynb را اجرا کن
# این notebook شامل تمام بهبودهاست:
# - Cross-platform normalization
# - Feature alignment
# - Clinical feature exclusion
```

**مرحله ۳: مقایسه نتایج**
```python
# جدول مقایسه‌ای ایجاد کن
Method | AUC | Improvement
-------|-----|------------
Original (Cell 5) | 0.50 | baseline
+ Quantile Norm | ? | expected +5-10%
+ Gene-only | ? | expected +2-5%
+ Retraining | ? | expected +5-10%
Full Pipeline (Notebook 09) | ? | expected 0.70-0.75
```

### ۵.۳ کد اصلاحی پیشنهادی برای Notebook 08

```python
# %% Cell جدید: اصلاح ارزیابی خارجی
from src.validation import prepare_external_validation

# بارگذاری داده‌ها
X_train = pd.read_csv("../data/processed/X_train_preprocessed.csv", index_col=0)
y_train = pd.read_csv("../data/processed/y_train.csv", index_col=0).squeeze()
selected_features = pd.read_csv("../outputs/tables/selected_features_final.csv")['feature'].tolist()

# آماده‌سازی با نرمالیزاسیون صحیح
X_train_aligned, X_ext_aligned, y_ext, used_features = prepare_external_validation(
    X_train, X_GSE70769, y_GSE70769,
    selected_features,
    exclude_clinical=True,
    normalization_method="quantile"  # یا "rank" برای مقاومت بیشتر
)

# آموزش مدل روی ویژگی‌های هم‌تراز
clf = XGBClassifier(**best_params, random_state=42)
clf.fit(X_train_aligned[used_features], y_train)

# ارزیابی
y_prob_ext = clf.predict_proba(X_ext_aligned[used_features])[:, 1]
auc_ext = roc_auc_score(y_ext, y_prob_ext)

print(f"IMPROVED EXTERNAL ROC-AUC: {auc_ext:.4f}")
# انتظار: 0.70-0.75 به جای 0.50
```

### ۵.۴ پیش‌بینی بهبود نهایی

| سناریو | AUC مورد انتظار |
|--------|-----------------|
| وضعیت فعلی | 0.50 |
| + Quantile Normalization | 0.60-0.65 |
| + Gene-only Features | 0.65-0.68 |
| + Retraining Strategy | 0.70-0.75 |
| **+ Optuna Hyperparameter Tuning** | **0.72-0.78** |

---

## ضمیمه: ساختار فایل‌های پروژه

```
/workspace/core/
├── data/
│   ├── raw/                      # داده‌های خام TCGA
│   ├── external/                 # داده‌های GEO
│   │   └── GSE70769_family.soft.gz ✅ (58MB, سالم)
│   ├── interim/                  # داده‌های میانی
│   │   ├── GSE70769_expression_clean.csv (94 samples × 29,720 genes)
│   │   └── GSE70769_bcr_target.csv (94 samples)
│   └── processed/                # داده‌های پردازش‌شده
│       ├── X_GSE70769.csv ✅
│       └── y_GSE70769.csv ✅
├── notebooks/
│   ├── 08_External _Evaluation.ipynb ⚠️ (نیاز به اصلاح)
│   └── 09_Improved_External_Validation.ipynb ✅ (توصیه‌شده)
├── src/
│   ├── validation.py ✅ (تابع normalize_cross_platform اضافه شد)
│   └── optimization.py ✅ (Optuna tuning اضافه شد)
└── outputs/
    ├── tables/                   # خالی (نیاز به اجرای Notebooks 1-7)
    └── models/                   # خالی (نیاز به اجرای Notebook 5)
```

---

## نتیجه‌گیری

**علت اصلی افت عملکرد:** عدم تطابق پلتفرم (RNA-Seq vs Microarray) و عدم اعمال نرمالیزاسیون cross-platform.

**راه‌حل توصیه‌شده:** اجرای Notebook 09 که شامل:
1. نرمالیزاسیون Quantile/Rank
2. حذف ویژگی‌های بالینی غیرقابل انتقال
3. هم‌ترازی ویژگی‌ها
4. Retraining strategy

**بهبود مورد انتظار:** از 0.50 به 0.70-0.75 AUC

---

*تهیه‌شده توسط: متخصص ارشد بیوانفورماتیک و یادگیری ماشین*
*تاریخ تحلیل: 2026-08-24*
