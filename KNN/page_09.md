
## 📖 صفحه ۹: تحلیل همبستگی ویژگی‌ها با Heatmap

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 چرا همبستگی مهم است؟

ویژگی‌ها همیشه مستقل از هم نیستند. برای مثال:

* BMI (شاخص توده بدنی) ممکن است با Insulin (انسولین خون) مرتبط باشد.
* Glucose معمولاً با Age همبستگی دارد چون با افزایش سن، احتمال افزایش قند خون بیشتر است.

📌 اگر دو ویژگی همبستگی بالایی داشته باشند، وجود هر دو در مدل KNN ممکن است غیرضروری باشد.

---

### 🔹 رسم ماتریس همبستگی و Heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlation matrix
corr_matrix = data.corr()

# Plot heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Diabetes Dataset")
plt.show()
```

---

### 🔹 تحلیل Heatmap

📌 نتایج معمول:

* Glucose بیشترین همبستگی مثبت را با Outcome دارد (یعنی مهم‌ترین ویژگی).
* BMI و DiabetesPedigreeFunction نیز همبستگی نسبتاً خوبی با Outcome نشان می‌دهند.
* بعضی ویژگی‌ها مثل SkinThickness و Insulin همبستگی ضعیفی با Outcome دارند و می‌توانند حذف شوند.

---

### 🔹 نکات کلیدی از همبستگی

1. **Glucose** → شاخص اصلی تشخیص دیابت.
2. **BMI** → نشان‌دهنده اضافه‌وزن، عامل قوی در بروز دیابت.
3. **Age** → افراد مسن‌تر احتمال بیشتری برای دیابت دارند.
4. **SkinThickness و BloodPressure** → همبستگی کمی با Outcome دارند و می‌توانند کمتر در تصمیم‌گیری نقش داشته باشند.

---

### 🔹 نتیجه‌گیری

تحلیل همبستگی نشان می‌دهد که استفاده از ویژگی‌های قوی (Glucose، BMI، Age) می‌تواند دقت مدل را افزایش دهد. از سوی دیگر، حذف ویژگی‌های ضعیف‌تر باعث سبک‌تر شدن مدل و کاهش نویز می‌شود.

