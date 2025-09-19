
## 📖 صفحه ۱۰: تحلیل همبستگی ویژگی‌ها و جمع‌بندی نهایی

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 اهمیت بررسی همبستگی

در پروژه‌های داده‌کاوی، بررسی **ارتباط بین ویژگی‌ها** به ما کمک می‌کند که:

* متوجه شویم کدام ویژگی‌ها تکراری یا وابسته هستند.
* بفهمیم چه ویژگی‌هایی تأثیر بیشتری روی برچسب (Spam/Ham) دارند.
* از داده‌های پر噪 (noise) یا ویژگی‌های بی‌اثر دوری کنیم.

---

### 🔹 آماده‌سازی داده برای همبستگی

از آنجا که متن‌ها به صورت بردارهای TF-IDF هستند (که بسیار بزرگ‌اند)، در اینجا برای سادگی فقط ویژگی‌های ساده‌تر مثل طول ایمیل یا تعداد حروف بزرگ (Uppercase letters) را بررسی می‌کنیم.

```python
# Calculate the number of uppercase letters in the 'text' column of the DataFrame 'data'
# and store the result in a new column 'uppercase_count'.
data['uppercase_count'] = data['text'].apply(lambda x: sum(1 for c in x if c.isupper()))

# Select specific features: 'email_length', 'uppercase_count', and 'label' from the DataFrame
features_corr = data[['email_length', 'uppercase_count', 'label']]

# Compute the correlation matrix for the selected features to analyze the relationships
# between 'email_length', 'uppercase_count', and 'label'.
corr_matrix = features_corr.corr()

# Print the resulting correlation matrix.
print(corr_matrix)
```

---

### 🔹 رسم Heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.show()
```

---

### 🔹 تفسیر Heatmap

📌 نتایج معمولاً به این شکل‌اند:

* ویژگی **email\_length** (طول ایمیل) همبستگی منفی با اسپم دارد → یعنی هرچه ایمیل کوتاه‌تر باشد، احتمال اسپم بودن بیشتر است.
* ویژگی **uppercase\_count** (تعداد حروف بزرگ) همبستگی مثبت با اسپم دارد → چون در اسپم‌ها معمولاً از جملاتی مثل **WIN NOW! FREE OFFER!** استفاده می‌شود.
* ارتباط بین ویژگی‌ها باهم نیز دیده می‌شود (مثلاً ایمیل‌های طولانی‌تر معمولاً تعداد حروف بزرگ بیشتری دارند).

---

### 🔹 جمع‌بندی پروژه

در این پروژه:

1. الگوریتم **Naïve Bayes** را معرفی کردیم و فرمول ریاضی آن را با یک مثال کوچک توضیح دادیم.
2. داده‌ی اسپم ایمیل را آماده‌سازی و پیش‌پردازش کردیم.
3. مدل Naïve Bayes را با **CountVectorizer** و سپس با **TF-IDF** آموزش دادیم.
4. عملکرد مدل را با معیارهای **Accuracy، Precision، Recall و F1-score** ارزیابی کردیم.
5. مهم‌ترین کلمات مؤثر در اسپم بودن را استخراج و با **Bar Plot** نمایش دادیم.
6. ویژگی‌های ساده مثل طول ایمیل را با **Boxplot** تحلیل کردیم.
7. ارتباط بین ویژگی‌ها را با **Heatmap** بررسی کردیم.

📌 نتیجه:
مدل Naïve Bayes با وجود سادگی، دقت بالایی در تشخیص اسپم دارد و با تحلیل داده‌های جانبی مثل طول ایمیل و تعداد حروف بزرگ، می‌توانیم آن را حتی بهتر کنیم.

---


