## 📖 صفحه ۳: معرفی دیتاست اسپم ایمیل و تحلیل اولیه داده‌ها

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 معرفی دیتاست اسپم ایمیل

برای پروژه‌ی عملی تشخیص اسپم، از یک دیتاست واقعی استفاده می‌کنیم که شامل مجموعه‌ای از ایمیل‌های **Spam** و **Ham** است. این دیتاست معمولاً شامل ستون‌هایی است که نشان می‌دهند آیا کلمات خاصی در ایمیل وجود دارند یا خیر. به‌طور نمونه می‌توان به دیتاست معروف **SpamBase** از مخزن UCI Machine Learning اشاره کرد.

در این دیتاست:

* هر سطر یک ایمیل است.
* هر ستون یک ویژگی (Feature) مربوط به وجود یا فراوانی کلمات خاص (مثل free, win, money) در ایمیل است.
* ستون آخر برچسب (Label) است:

  * ۱ = ایمیل اسپم (Spam)
  * ۰ = ایمیل عادی (Ham)

---

### 🔹 بارگذاری و مشاهده داده‌ها در پایتون

```python
# Importing essential libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (example: UCI Spambase or custom CSV file)
# Replace 'spam.csv' with actual dataset path
data = pd.read_csv("spam.csv")

# Show first 5 rows
print(data.head())
```

---

### 🔹 تحلیل اولیه داده‌ها

قبل از اینکه مدل بسازیم، باید داده‌ها را بهتر بشناسیم. پرسش‌های کلیدی در تحلیل اولیه عبارتند از:

* چند درصد ایمیل‌ها اسپم هستند و چند درصد عادی؟
* کدام ویژگی‌ها بیشتر در اسپم دیده می‌شوند؟
* آیا بین ویژگی‌ها همبستگی وجود دارد؟

---

### 🔹 توزیع برچسب‌ها (Spam vs Ham)

```python
# Count of spam and ham
plt.figure(figsize=(6,4))
data['label'].value_counts().plot(kind='bar', color=['skyblue','salmon'])
plt.xticks([0,1], ['Ham','Spam'], rotation=0)
plt.title("Distribution of Ham vs Spam Emails")
plt.xlabel("Email Type")
plt.ylabel("Count")
plt.show()
```

📌 این نمودار نشان می‌دهد که چه نسبتی از ایمیل‌ها اسپم هستند. معمولاً تعداد ایمیل‌های عادی بیشتر از اسپم‌هاست.

---

### 🔹 بررسی ارتباط بین ویژگی‌ها

یکی از روش‌های مهم تحلیل داده، محاسبه‌ی همبستگی (Correlation) بین ویژگی‌هاست.

```python
# Compute correlation matrix
corr = data.corr()

# Heatmap for correlation
plt.figure(figsize=(12,8))
sns.heatmap(corr, cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap of Features")
plt.show()
```

📌 این Heatmap به ما کمک می‌کند ببینیم کدام ویژگی‌ها با هم همبستگی بالا دارند و کدام ویژگی‌ها می‌توانند در پیش‌بینی اسپم مفیدتر باشند.

---

### 🔹 بررسی یک ویژگی خاص با Boxplot

برای مثال، بررسی می‌کنیم توزیع کلمه‌ی **Free** در ایمیل‌های اسپم و عادی چطور است.

```python
# Boxplot for word "Free" frequency
plt.figure(figsize=(6,4))
sns.boxplot(x="label", y="Free", data=data)
plt.xticks([0,1], ['Ham','Spam'])
plt.title("Boxplot of word 'Free' in Ham vs Spam Emails")
plt.show()
```

📌 اگر این ویژگی در اسپم‌ها مقدار بیشتری داشته باشد، می‌تواند یک شاخص مهم برای مدل باشد.

---

در این صفحه، با دیتاست آشنا شدیم و تحلیل‌های اولیه مثل توزیع داده‌ها، همبستگی ویژگی‌ها، و یک Boxplot نمونه را بررسی کردیم. در گام بعدی، به‌طور سیستماتیک ویژگی‌های تأثیرگذارتر را انتخاب می‌کنیم.

