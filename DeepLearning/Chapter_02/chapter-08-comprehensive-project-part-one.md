# فصل ۸ 🏗️ پروژه‌ی جامع بخش اول: از صفر تا یک مدل آماده‌ی صنعتی

## 🎯 اهداف فصل

این فصل متفاوت‌ترین فصل از آنچه تاکنون خوانده‌اید است. در اینجا مفهوم جدیدی معرفی نمی‌شود؛ در عوض، تمام آنچه در هفت فصل گذشته آموختیم را در قالب یک پروژه‌ی کامل، واقعی و چندمرحله‌ای با هم ترکیب می‌کنیم. هدف این است که در پایان این فصل، تجربه‌ی ساخت یک مدل یادگیری عمیق واقعی از صفر تا انتها را داشته باشید؛ از بارگذاری و تحلیل اولیه‌ی داده گرفته تا مهندسی ویژگی، ساخت معماری مدل، آموزش با تمام بهترین تکنیک‌ها، ارزیابی جامع و در نهایت ذخیره‌ی مدل برای استفاده‌ی آینده. این فصل را به‌عنوان یک پروژه‌ی رزومه‌ای واقعی در نظر بگیرید.

---

# 📋 تعریف مسئله: پیش‌بینی ریزش مشتری

مسئله‌ای که در این فصل حل می‌کنیم، پیش‌بینی ریزش مشتری (Customer Churn Prediction) در یک شرکت مخابراتی است. این مسئله در دنیای واقعی بسیار رایج و اقتصادی است: هزینه‌ی جذب یک مشتری جدید معمولاً پنج تا هفت برابر هزینه‌ی نگه‌داشتن یک مشتری فعلی است، پس اگر بتوانیم پیش از آن‌که مشتری اشتراکش را لغو کند، او را شناسایی کنیم، می‌توانیم با ارائه‌ی پیشنهادهای ویژه او را حفظ کنیم. این دقیقاً همان نوع مسئله‌ای است که در مصاحبه‌های شغلی یادگیری ماشین و یادگیری عمیق از شما خواسته می‌شود.

ما از دیتاست Telco Customer Churn استفاده می‌کنیم که شامل اطلاعات واقعی بیش از هفت هزار مشتری از جمله مدت اشتراک، نوع قرارداد، هزینه‌ی ماهانه، خدمات استفاده‌شده و وضعیت ریزش آن‌هاست. این یک مسئله‌ی طبقه‌بندی دوکلاسه با توزیع نامتوازن است (تنها حدود بیست و شش درصد مشتریان ریزش کرده‌اند) که درک مفاهیم فصل هفت را در عمل به کار می‌برد.

---

# 🔍 مرحله‌ی اول: بارگذاری و تحلیل اکتشافی داده

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)

# بارگذاری دیتاست از مخزن آنلاین
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# بررسی اولیه ساختار داده
print(f"ابعاد داده: {df.shape}")
print(f"\nنوع داده هر ستون:\n{df.dtypes}")
print(f"\nتعداد مقادیر گمشده:\n{df.isnull().sum().sum()}")

# توزیع کلاس هدف
churn_dist = df['Churn'].value_counts(normalize=True) * 100
print(f"\nتوزیع کلاس هدف:")
print(f"مشتریان باقی‌مانده: {churn_dist['No']:.1f}%")
print(f"مشتریان ریزش‌کرده: {churn_dist['Yes']:.1f}%")
```

بررسی اولیه‌ی داده نشان می‌دهد که این دیتاست دارای هفت هزار و چهل و سه ردیف و بیست و یک ستون است که شامل ترکیبی از ویژگی‌های عددی (مانند مدت اشتراک و هزینه‌ی ماهانه) و ویژگی‌های دسته‌ای (مانند نوع قرارداد و روش پرداخت) است. مهم‌ترین نکته‌ای که از تحلیل اولیه به دست می‌آید، نامتوازن بودن کلاس‌هاست: تنها حدود بیست و شش درصد از مشتریان ریزش کرده‌اند. این موضوع تأثیر مستقیمی بر انتخاب معیار ارزیابی و گاهی حتی روش آموزش ما خواهد داشت.

---

# 🔧 مرحله‌ی دوم: پیش‌پردازش و مهندسی ویژگی

```python
# حذف ستون شناسه که اطلاعاتی ندارد
df = df.drop('customerID', axis=1)

# تبدیل TotalCharges به عدد و پر کردن مقادیر گمشده
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# رمزگذاری متغیر هدف
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

# جداسازی ویژگی‌های دسته‌ای و عددی
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# رمزگذاری ویژگی‌های دسته‌ای با One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# جداسازی ویژگی‌ها از متغیر هدف
X = df_encoded.drop('Churn', axis=1).values.astype(np.float32)
y = df_encoded['Churn'].values.astype(np.float32)

print(f"تعداد ویژگی‌های نهایی پس از رمزگذاری: {X.shape[1]}")

# تقسیم سه‌گانه: آموزش، اعتبارسنجی و آزمون
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
)

# استانداردسازی فقط بر اساس داده آموزشی
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# تبدیل به تنسور
def to_tensor(X, y):
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    )

X_train_t, y_train_t = to_tensor(X_train, y_train)
X_val_t, y_val_t = to_tensor(X_val, y_val)
X_test_t, y_test_t = to_tensor(X_test, y_test)

print(f"اندازه مجموعه آموزش: {X_train_t.shape[0]} نمونه")
print(f"اندازه مجموعه اعتبارسنجی: {X_val_t.shape[0]} نمونه")
print(f"اندازه مجموعه آزمون: {X_test_t.shape[0]} نمونه")
```

یک نکته‌ی مهم در این مرحله که باید به آن توجه کنید، استفاده از پارامتر `stratify=y` در تقسیم داده است. این پارامتر تضمین می‌کند که نسبت کلاس‌های مثبت و منفی در هر سه مجموعه‌ی آموزش، اعتبارسنجی و آزمون یکسان باشد. بدون این پارامتر، ممکن است به‌طور تصادفی یکی از مجموعه‌ها مشتریان ریزش‌کرده‌ی بسیار کمتر یا بیشتری داشته باشد که مقایسه‌ی عملکرد مدل را دشوار می‌کند.

---

# 🏗️ مرحله‌ی سوم: طراحی معماری مدل

```python
class ChurnPredictor(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        # بلوک اول: استخراج ویژگی‌های اولیه
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        # بلوک دوم: فشرده‌سازی تدریجی
        self.block2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # بلوک سوم: نمایش انتزاعی نهایی
        self.block3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # لایه خروجی با Sigmoid برای خروجی احتمالاتی
        self.output = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.output(x)

input_dim = X_train_t.shape[1]
model = ChurnPredictor(input_dim)

# محاسبه تعداد کل پارامترهای قابل‌یادگیری
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"تعداد کل پارامترهای قابل‌یادگیری: {total_params:,}")
```

طراحی این معماری بر اساس اصل «فشرده‌سازی تدریجی» انجام شده است: از یکصد و بیست و هشت نورون در بلوک اول به شصت و چهار در بلوک دوم و سپس به سی و دو در بلوک سوم می‌رسیم. این الگو باعث می‌شود شبکه مجبور شود اطلاعات مهم را در فضای کوچک‌تری فشرده کند و از این طریق ویژگی‌های انتزاعی‌تر و معنادارتری یاد بگیرد. همچنین نرخ Dropout در لایه‌های ابتدایی بیشتر (چهل درصد) و در لایه‌های انتهایی کمتر (بیست درصد) است، چراکه لایه‌های اولیه با داده‌ی خام در تماس هستند و بیشتر در معرض حفظ کردن نویز.

---

# 🚀 مرحله‌ی چهارم: آموزش با تمام بهترین تکنیک‌ها

```python
# مدیریت عدم‌تعادل کلاس‌ها با وزن‌دهی به تابع هزینه
pos_weight = torch.tensor(
    [(y_train == 0).sum() / (y_train == 1).sum()],
    dtype=torch.float32
)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-3
)

# زمان‌بند نرخ یادگیری: کاهش تدریجی نرخ یادگیری
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=10, factor=0.5
)

# آموزش با Early Stopping خودکار
best_val_loss = float('inf')
best_model_weights = None
patience_counter = 0
patience_limit = 20
history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

for epoch in range(300):

    model.train()
    optimizer.zero_grad()

    # برای BCEWithLogitsLoss باید خروجی بدون Sigmoid باشد
    outputs = model.block1(X_train_t)
    outputs = model.block2(outputs)
    outputs = model.block3(outputs)
    outputs = model.output[0](outputs)  # فقط لایه Linear، بدون Sigmoid

    train_loss = criterion(outputs, y_train_t)
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_out = model.block1(X_val_t)
        val_out = model.block2(val_out)
        val_out = model.block3(val_out)
        val_logits = model.output[0](val_out)
        val_loss = criterion(val_logits, y_val_t).item()

        val_probs = torch.sigmoid(val_logits).numpy().flatten()
        val_preds = (val_probs > 0.5).astype(int)
        val_f1 = f1_score(y_val.astype(int), val_preds, zero_division=0)

    scheduler.step(val_loss)
    history['train_loss'].append(train_loss.item())
    history['val_loss'].append(val_loss)
    history['val_f1'].append(val_f1)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience_limit:
        print(f"Early Stopping در دوره {epoch}: خطای اعتبارسنجی برای {patience_limit} دوره بهبود نیافت.")
        break

    if epoch % 50 == 0:
        print(f"دوره {epoch}: خطای آموزش {train_loss.item():.4f} | خطای اعتبارسنجی {val_loss:.4f} | F1 اعتبارسنجی {val_f1:.3f}")
```

در این مرحله چند تکنیک پیشرفته به کار رفته که ارزش توضیح دارند. نخست، برای مدیریت عدم‌تعادل کلاس‌ها از `BCEWithLogitsLoss` با `pos_weight` استفاده شده است؛ این وزن به مدل می‌گوید که اشتباه در شناسایی مشتریان ریزش‌کرده (کلاس اقلیت) را سنگین‌تر جریمه کند. دوم، از بهینه‌ساز AdamW که نسخه‌ی بهبودیافته‌ی Adam است استفاده شده که Weight Decay را به‌درستی‌تری اعمال می‌کند. سوم، از `ReduceLROnPlateau` برای کاهش خودکار نرخ یادگیری در صورت رکود استفاده شده، که آموزش را پایدارتر می‌کند. چهارم، Early Stopping با patience برابر بیست پیاده‌سازی شده که از اتلاف منابع محاسباتی جلوگیری می‌کند.

---

# 📊 مرحله‌ی پنجم: ارزیابی جامع نهایی

```python
# بارگذاری بهترین مدل
model.load_state_dict(best_model_weights)
model.eval()

with torch.no_grad():
    test_out = model.block1(X_test_t)
    test_out = model.block2(test_out)
    test_out = model.block3(test_out)
    test_logits = model.output[0](test_out)
    test_probs = torch.sigmoid(test_logits).numpy().flatten()

test_preds = (test_probs > 0.5).astype(int)
y_test_int = y_test.astype(int)

# گزارش کامل ارزیابی
print("=" * 50)
print("گزارش کامل ارزیابی مدل روی مجموعه آزمون")
print("=" * 50)
print(classification_report(
    y_test_int, test_preds,
    target_names=['باقی‌مانده', 'ریزش‌کرده']
))

cm = confusion_matrix(y_test_int, test_preds)
auc = roc_auc_score(y_test_int, test_probs)

print(f"ماتریس درهم‌ریختگی:\n{cm}")
print(f"\nمساحت زیر منحنی ROC (AUC): {auc:.4f}")
print(f"\nتفسیر تجاری:")
print(f"از هر ۱۰۰ مشتری که قصد ریزش دارند، مدل {int(f1_score(y_test_int, test_preds)*100)} نفر را شناسایی می‌کند.")

# ذخیره مدل برای استفاده آینده
torch.save({
    'model_state_dict': best_model_weights,
    'scaler_mean': scaler.mean_,
    'scaler_std': scaler.scale_,
    'input_dim': input_dim,
    'feature_names': list(df_encoded.drop('Churn', axis=1).columns)
}, 'churn_model.pt')

print("\nمدل با موفقیت ذخیره شد: churn_model.pt")
```

---

# 🔄 مرحله‌ی ششم: استفاده از مدل ذخیره‌شده برای پیش‌بینی جدید

```python
def predict_churn(new_customer_data, model_path='churn_model.pt'):

    # بارگذاری مدل و پارامترهای پیش‌پردازش
    checkpoint = torch.load(model_path)

    model = ChurnPredictor(checkpoint['input_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # اعمال همان استانداردسازی مرحله آموزش
    X_new = (new_customer_data - checkpoint['scaler_mean']) / checkpoint['scaler_std']
    X_tensor = torch.tensor(X_new, dtype=torch.float32)

    with torch.no_grad():
        out = model.block1(X_tensor)
        out = model.block2(out)
        out = model.block3(out)
        logit = model.output[0](out)
        prob = torch.sigmoid(logit).item()

    result = "ریزش محتمل" if prob > 0.5 else "ماندن محتمل"
    print(f"احتمال ریزش: {prob*100:.1f}% — {result}")
    return prob

print("مدل آماده‌ی استفاده برای پیش‌بینی مشتریان جدید است.")
```

---

# ❓ سؤالات تستی

### سؤال ۱

در این پروژه چرا از پارامتر `pos_weight` در تابع هزینه استفاده شد؟

الف) برای افزایش سرعت آموزش

ب) برای جبران عدم‌تعادل کلاس‌ها و دادن وزن بیشتر به خطاهای کلاس اقلیت (مشتریان ریزش‌کرده)

ج) برای کاهش تعداد پارامترهای مدل

د) برای جلوگیری از بیش‌برازش

### سؤال ۲

دلیل ذخیره‌ی `scaler_mean` و `scaler_std` به‌همراه مدل در فایل `churn_model.pt` چیست؟

الف) چون این مقادیر بخشی از پارامترهای شبکه‌ی عصبی هستند

ب) برای این‌که در زمان استفاده از مدل روی داده‌های جدید، دقیقاً همان تبدیل پیش‌پردازش مرحله‌ی آموزش اعمال شود و داده‌های جدید با همان مقیاس وارد مدل شوند

ج) چون PyTorch بدون این مقادیر قادر به بارگذاری مدل نیست

د) برای محاسبه‌ی تعداد پارامترهای مدل

---

## ✅ پاسخ‌ها

**پاسخ سؤال ۱: گزینه ب**

وقتی توزیع کلاس‌ها نامتوازن است، مدل بدون وزن‌دهی تمایل پیدا می‌کند که اکثر پیش‌بینی‌هایش را به کلاس اکثریت اختصاص دهد، چراکه این کار خطای کمتری تولید می‌کند. با استفاده از `pos_weight`، جریمه‌ی اشتباه در شناسایی کلاس اقلیت بزرگ‌تر می‌شود و مدل مجبور می‌شود توجه بیشتری به آن‌ها داشته باشد.

**پاسخ سؤال ۲: گزینه ب**

یک مدل آموزش‌دیده تنها زمانی پیش‌بینی‌های معناداری می‌دهد که ورودی‌هایش با همان مقیاسی که در طول آموزش داشتند وارد شوند. اگر پارامترهای scaler ذخیره نشوند، در زمان استقرار مدل مجبور می‌شویم یا داده‌ی آموزشی را نگه‌داری کنیم (که معمولاً ممکن نیست) یا با مقیاس‌های اشتباه پیش‌بینی کنیم.

---

# 📝 خلاصه فصل

در این فصل یک پروژه‌ی کامل یادگیری عمیق را از ابتدا تا انتها پیاده‌سازی کردیم. مسئله‌ی پیش‌بینی ریزش مشتری را به‌عنوان بستر انتخاب کردیم که هم از نظر تجاری کاربردی است و هم چالش‌های واقعی مانند عدم‌تعادل کلاس‌ها را در خود دارد. در شش مرحله‌ی مجزا، تحلیل اولیه‌ی داده، پیش‌پردازش و رمزگذاری، طراحی معماری مدل، آموزش با تکنیک‌های پیشرفته، ارزیابی جامع و در نهایت ذخیره و استفاده‌ی مجدد از مدل را تجربه کردیم. این پروژه ترکیبی از تمام مفاهیم هفت فصل گذشته است و می‌تواند به‌عنوان یک نمونه‌کار (Portfolio) واقعی در رزومه‌ی شما قرار گیرد.

---

# 🎯 تمرین‌های پایان فصل

۱. آستانه‌ی تصمیم‌گیری را از پنجاه درصد به چهل درصد تغییر دهید و ببینید تأثیر آن بر Recall مشتریان ریزش‌کرده چیست.

۲. معماری مدل را با اضافه کردن یک بلوک چهارم (شانزده نورون) گسترش دهید و نتایج را مقایسه کنید.

۳. به‌جای One-Hot Encoding، از Target Encoding برای ویژگی‌های دسته‌ای استفاده کنید و اثر آن بر عملکرد مدل را بسنجید.

۴. از SMOTE (روش نمونه‌برداری مصنوعی از کلاس اقلیت) به‌جای `pos_weight` برای مدیریت عدم‌تعادل استفاده کنید.

۵. تابع `predict_churn` را گسترش دهید تا بتواند دسته‌ای از مشتریان جدید را به‌یک‌باره پردازش کند و نتایج را در قالب یک DataFrame پانداس برگرداند.

---

با پایان این فصل، بخش اول کتاب که به مبانی شبکه‌های عصبی اختصاص داشت، به پایان رسید. در **بخش دوم** که از فصل ۹ آغاز می‌شود، وارد دنیای بینایی ماشین و شبکه‌های کانولوشنی می‌شویم؛ معماری‌هایی که انقلابی در توانایی ماشین‌ها برای درک و پردازش تصویر ایجاد کردند.
