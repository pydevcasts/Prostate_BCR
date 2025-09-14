## 📖 صفحه ۳: استانداردسازی و آماده‌سازی داده‌ها

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 چرا استانداردسازی مهم است؟

ویژگی‌های دیتاست Wine در مقیاس‌های مختلف قرار دارند:

* مقدار **Alcohol** معمولاً بین ۱۲ تا ۱۵ است.
* مقدار **Magnesium** می‌تواند تا ۱۶۰ برسد.
* مقدار **Proline** حتی به بالای ۱۰۰۰ هم می‌رسد.

اگر این ویژگی‌ها بدون استانداردسازی به مدل داده شوند، ویژگی‌هایی با مقادیر بزرگ‌تر (مثل Proline) تأثیر بیشتری در آموزش خواهند داشت و این باعث **انحراف مدل** می‌شود.

---

### 🔹 فرمول استانداردسازی

$$
X_{scaled} = \frac{X - \mu}{\sigma}
$$

که در آن:

* $\mu$: میانگین ویژگی
* $\sigma$: انحراف معیار ویژگی

📌 بعد از استانداردسازی، همه ویژگی‌ها میانگین صفر و انحراف معیار یک خواهند داشت.

---

### 🔹 پیاده‌سازی استانداردسازی با Scikit-Learn

```python
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Mean after scaling (approx.):", X_scaled.mean(axis=0)[:5])
print("Std after scaling (approx.):", X_scaled.std(axis=0)[:5])
```

📊 خروجی نشان خواهد داد که میانگین تقریباً ۰ و انحراف معیار تقریباً ۱ است.

---

### 🔹 بررسی دوباره داده‌ها با Boxplot بعد از استانداردسازی

```python
plt.figure(figsize=(15,8))
sns.boxplot(data=pd.DataFrame(X_scaled, columns=wine.feature_names))
plt.xticks(rotation=90)
plt.title("Boxplot of Standardized Wine Features")
plt.show()
```

📌 حالا همه ویژگی‌ها در یک مقیاس مشابه قرار گرفتند. این موضوع برای الگوریتم‌هایی مثل KNN و SVM بسیار حیاتی است.

---

### 🔹 آماده‌سازی داده‌ها برای Cross Validation

برای ادامه کار، باید داده‌ها را به دو بخش تقسیم کنیم:

* ویژگی‌ها (X\_scaled)
* برچسب‌ها (y)

```python
from sklearn.model_selection import KFold

# Define K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Number of splits:", kf.get_n_splits())
```

📌 در اینجا ما از **۵-Fold Cross Validation** استفاده می‌کنیم. یعنی داده‌ها ۵ بار تقسیم می‌شوند و هر بار مدل روی بخشی آموزش داده می‌شود و روی بخش دیگر تست می‌شود.

---

### 🔹 جمع‌بندی صفحه ۳

* داده‌ها را استانداردسازی کردیم تا همه ویژگی‌ها در یک مقیاس قرار بگیرند.
* با Boxplot بررسی کردیم که ویژگی‌ها بعد از استانداردسازی نرمال‌تر شدند.
* برای مراحل بعدی، یک **KFold Cross Validation با ۵ بخش** آماده کردیم.
