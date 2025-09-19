## 📖 صفحه ۸: پیش‌پردازش داده‌ها برای رگرسیون لجستیک

✍️ نویسنده: سیامک عباس‌نژاد

پیش‌پردازش داده‌ها یکی از مراحل کلیدی در علم داده است. اگر داده‌ها کیفیت خوبی نداشته باشند، حتی بهترین الگوریتم‌ها هم عملکرد مطلوبی نخواهند داشت. دیتاست دیابت هم از این قاعده مستثنی نیست و نیاز به آماده‌سازی دقیق دارد.

---

### 🔹 شناسایی مقادیر صفر غیرواقعی

در برخی ویژگی‌ها مثل **BloodPressure، BMI، Insulin و SkinThickness** مقادیر صفر به‌معنای واقعی صفر نیستند. این مقادیر در واقع نشان‌دهنده‌ی داده‌ی گمشده‌اند. برای شناسایی تعداد مقادیر صفر در هر ستون می‌توانیم کد زیر را اجرا کنیم:

```python
# Count zero values in each column
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    print(col, (data[col] == 0).sum())
```

📌 نتیجه: مشاهده می‌شود که تعداد زیادی مقدار صفر در ستون‌های **Insulin** و **SkinThickness** وجود دارد که باید اصلاح شوند.

---

### 🔹 جایگزینی مقادیر صفر با میانه (Median Imputation)

یکی از روش‌های متداول برای اصلاح داده‌های گمشده، جایگزینی آن‌ها با میانه یا میانگین است. در این پروژه از **میانه (Median)** استفاده می‌کنیم چون نسبت به داده‌های پرت (Outliers) مقاوم‌تر است.

```python
# Replace zeros with median values
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    median = data[col].median()
    data[col] = data[col].replace(0, median)
```

---

### 🔹 مقیاس‌بندی داده‌ها (Feature Scaling)

از آنجایی که ویژگی‌ها در مقیاس‌های متفاوت قرار دارند (مثلاً Age در بازه‌ی ۲۰ تا ۸۰ و Glucose در بازه‌ی ۰ تا ۲۰۰)، باید آن‌ها را نرمال‌سازی کنیم. الگوریتم رگرسیون لجستیک به مقیاس داده‌ها حساس است.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop("Outcome", axis=1))

X = pd.DataFrame(scaled_data, columns=data.columns[:-1])
y = data["Outcome"]
```

---

### 🔹 تقسیم داده‌ها به آموزش و تست

برای ارزیابی مدل باید داده‌ها را به دو بخش تقسیم کنیم:

* داده‌های آموزش (Training set) برای یادگیری مدل
* داده‌های تست (Test set) برای ارزیابی عملکرد مدل

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

📍 در این مرحله داده‌ها آماده‌ی استفاده هستند. در صفحه بعد (**صفحه ۸**) وارد بخش اصلی پروژه می‌شویم: **آموزش مدل رگرسیون لجستیک و ارزیابی اولیه‌ی آن**.


