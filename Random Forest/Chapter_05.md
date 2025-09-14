## 📖 صفحه ۵: تحلیل داده‌ها با Pairplot

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 چرا Pairplot؟

ابزار **Pairplot** در کتابخانه‌ی Seaborn این امکان را می‌دهد که:

* رابطه بین چند ویژگی را به صورت **Scatter Plot** دوبه‌دو مشاهده کنیم.
* توزیع تک‌متغیره هر ویژگی را روی قطر اصلی (Diagonal) ببینیم.
* با رنگ‌بندی بر اساس کلاس‌ها (Malignant vs Benign)، تفاوت‌ها را بهتر درک کنیم.

---

### 🔹 انتخاب چند ویژگی کلیدی

به دلیل اینکه دیتاست ۳۰ ویژگی دارد، نمایش همه ویژگی‌ها در Pairplot بسیار سنگین و ناخواناست.
بنابراین، ما چند ویژگی مهم (بر اساس همبستگی با Target) را انتخاب می‌کنیم:

* **mean radius**
* **mean texture**
* **mean perimeter**
* **mean area**
* **mean smoothness**

---

### 🔹 رسم Pairplot

```python
# Select important features
selected_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'target']

# Subset DataFrame
df_subset = df[selected_features]

# Pairplot
sns.pairplot(df_subset, hue="target", palette="Set1", diag_kind="kde")
plt.show()
```

---

### 🔹 تفسیر نمودارها

📊 نتایج Pairplot معمولاً این‌طور نشان داده می‌شود:

* **mean radius** و **mean area** برای Malignant و Benign مرزبندی خوبی دارند (تفکیک دو کلاس به وضوح دیده می‌شود).
* **mean texture** همپوشانی بیشتری دارد و به تنهایی قدرت تفکیک کمتری دارد.
* Scatterهای بین **mean radius** و **mean perimeter** نشان می‌دهند که این دو ویژگی همبستگی بالایی دارند (که در Heatmap هم دیده بودیم).

---

### 🔹 نتیجه‌گیری

تحلیل با Pairplot نشان داد:

* برخی ویژگی‌ها (مثل mean radius و mean area) قدرت بالایی برای جداسازی کلاس‌ها دارند.
* برخی دیگر (مثل mean texture) کمتر قابل اعتماد هستند.
* استفاده از چند ویژگی با هم می‌تواند قدرت تفکیک مدل را افزایش دهد.
