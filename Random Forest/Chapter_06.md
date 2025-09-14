## 📖 صفحه ۶: تحلیل داده‌ها با Violinplot

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 چرا Violinplot؟

Violinplot ترکیبی از **Boxplot** و **Density Plot (KDE)** است.
این نمودار علاوه بر نشان دادن:

* **میانه (Median)**
* **چارک‌ها (Quartiles)**
* **نقاط پرت (Outliers)**

همچنین **توزیع کامل داده‌ها** را نمایش می‌دهد. این باعث می‌شود که نسبت به Boxplot ساده، اطلاعات بسیار بیشتری در اختیار داشته باشیم.

---

### 🔹 رسم Violinplot برای چند ویژگی مهم

```python
# Violinplot for mean radius
plt.figure(figsize=(8,6))
sns.violinplot(x="target", y="mean radius", data=df, palette="muted")
plt.title("Violinplot of Mean Radius by Class")
plt.xticks([0,1], ['Benign', 'Malignant'])
plt.show()

# Violinplot for mean area
plt.figure(figsize=(8,6))
sns.violinplot(x="target", y="mean area", data=df, palette="muted")
plt.title("Violinplot of Mean Area by Class")
plt.xticks([0,1], ['Benign', 'Malignant'])
plt.show()
```

---

### 🔹 تفسیر نتایج

📊 در نمودارهای Violinplot معمولاً مشاهده می‌شود:

* در ویژگی **mean radius**:

  * کلاس Malignant (سرطانی) مقادیر بالاتری دارد.
  * کلاس Benign (خوش‌خیم) در مقادیر پایین‌تر متمرکز شده است.
* در ویژگی **mean area**:

  * Malignant مقادیر پراکنده‌تر و بزرگ‌تری دارد.
  * Benign دامنه محدودتری دارد.

---

### 🔹 مقایسه با Boxplot

* Boxplot تنها میانه و چارک‌ها را نشان می‌داد، اما Violinplot شکل توزیع داده‌ها را نیز به ما داد.
* مشخص شد که توزیع داده‌ها در Malignant معمولاً **چولگی به سمت مقادیر بالا** دارد.
* در Benign داده‌ها بیشتر در ناحیه‌ی پایین متمرکز هستند.

---

### 🔹 نتیجه‌گیری

* Violinplot نشان داد که **ویژگی‌های mean radius و mean area قدرت تفکیک بالایی بین دو کلاس دارند**.
* این نوع تحلیل تصویری به ما کمک می‌کند بفهمیم چرا مدل‌هایی مثل Random Forest می‌توانند این ویژگی‌ها را به عنوان شاخص‌های کلیدی انتخاب کنند.

