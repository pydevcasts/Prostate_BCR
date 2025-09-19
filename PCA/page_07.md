
---

## 📖 صفحه ۷: انتخاب تعداد مؤلفه‌ها با Explained Variance Ratio

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 چرا انتخاب تعداد مؤلفه‌ها مهم است؟

وقتی PCA اجرا می‌کنیم، می‌توانیم هر تعداد مؤلفه که بخواهیم انتخاب کنیم.
اما انتخاب درست تعداد مؤلفه‌ها اهمیت دارد:

* اگر تعداد مؤلفه‌ها خیلی کم باشد → اطلاعات مهم از دست می‌رود.
* اگر تعداد مؤلفه‌ها خیلی زیاد باشد → مدل همچنان پیچیده می‌ماند و سرعت کم می‌شود.

راه‌حل این است که بررسی کنیم هر مؤلفه چند درصد از واریانس داده را توضیح می‌دهد.

---

### 🔹 کد محاسبه Explained Variance Ratio

```python
# Apply PCA with all components
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

# Explained variance ratio
explained_var = pca_full.explained_variance_ratio_

# Plot explained variance ratio
plt.figure(figsize=(8,6))
plt.plot(range(1, len(explained_var)+1), explained_var.cumsum(), marker='o', linestyle='--', color='b')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance Ratio (Wine Dataset)")
plt.grid(True)
plt.show()
```

---

### 🔹 تحلیل نمودار

📌 معمولاً در دیتاست Wine:

* PC1 حدود **۳۶٪** از واریانس داده را توضیح می‌دهد.
* PC2 حدود **۱۹٪** دیگر را اضافه می‌کند.
* PC3 نیز حدود **۱۱٪** دیگر را پوشش می‌دهد.

به این ترتیب، فقط با ۳ مؤلفه اول تقریباً **۶۶٪** از اطلاعات کل داده حفظ می‌شود.

---

### 🔹 انتخاب تعداد مناسب مؤلفه‌ها

* اگر بخواهیم **Visualization** انجام دهیم، معمولاً ۲ یا ۳ مؤلفه کافی است.
* اگر بخواهیم **مدل‌سازی** دقیق‌تری داشته باشیم، ممکن است ۶ تا ۸ مؤلفه لازم باشد تا بیش از ۹۰٪ واریانس پوشش داده شود.

---

### 🔹 نتیجه این بخش

با بررسی **Explained Variance Ratio** مشخص شد که:

* مؤلفه‌های اولیه بیشترین نقش را در توضیح داده‌ها دارند.
* استفاده از ۲ یا ۳ مؤلفه برای Visualization و تحلیل داده مناسب است.
* برای ساخت مدل‌های یادگیری ماشین، معمولاً تعداد بیشتری مؤلفه نیاز داریم تا دقت بهینه به دست بیاید.
