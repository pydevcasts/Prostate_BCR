## 📖 فصل ۴: نمودارهای آماری

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. Histogram (هیستوگرام)

هیستوگرام توزیع داده‌ها را نشان می‌دهد و مشخص می‌کند داده‌ها در چه بازه‌هایی بیشتر متمرکز هستند.

```python
import matplotlib.pyplot as plt
import numpy as np

data = np.random.randn(1000)  # 1000 random numbers from normal distribution

plt.hist(data, bins=30, color="skyblue", edgecolor="black")
plt.title("Histogram Example")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

📌 این کد نشان می‌دهد داده‌ها بیشتر نزدیک به مقدار میانگین متمرکز شده‌اند.

---

### 🔹 ۲. Boxplot (نمودار جعبه‌ای)

Boxplot برای بررسی میانه، چارک‌ها و داده‌های پرت (Outliers) استفاده می‌شود.

```python
np.random.seed(42)
data1 = np.random.normal(0, 1, 100)
data2 = np.random.normal(5, 2, 100)

plt.boxplot([data1, data2], labels=["Group 1", "Group 2"])
plt.title("Boxplot Example")
plt.ylabel("Values")
plt.show()
```

📌 خطوط میانی جعبه نشان‌دهنده **میانه داده‌ها** هستند و نقاط بیرون جعبه، داده‌های پرت‌اند.

---

### 🔹 ۳. Violinplot (نمودار ویولن)

Violinplot ترکیبی از Boxplot و Kernel Density است و شکل توزیع داده‌ها را دقیق‌تر نشان می‌دهد.

```python
plt.violinplot([data1, data2], showmeans=True)
plt.title("Violinplot Example")
plt.ylabel("Values")
plt.xticks([1, 2], ["Group 1", "Group 2"])
plt.show()
```

📌 این نمودار علاوه بر نمایش میانه، توزیع کلی داده‌ها را هم نمایش می‌دهد.

---

### 🔹 ۴. Pie Chart پیشرفته (نمودار دایره‌ای با Explode)

```python
sizes = [40, 25, 20, 15]
labels = ["A", "B", "C", "D"]
explode = [0.1, 0, 0, 0]  # Highlight the first slice

plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90,
        colors=["gold", "lightblue", "lightgreen", "pink"], explode=explode, shadow=True)
plt.title("Advanced Pie Chart Example")
plt.show()
```

📌 بخش اول (A) از نمودار برجسته‌تر نمایش داده می‌شود.

---

### 🔹 نتیجه‌گیری فصل ۴

* **Histogram** → بررسی توزیع داده‌ها.
* **Boxplot** → مشاهده میانه، چارک‌ها و داده‌های پرت.
* **Violinplot** → نمایش شکل توزیع داده‌ها همراه با جزئیات.
* **Pie Chart پیشرفته** → نمایش درصد سهم دسته‌ها همراه با بخش برجسته (Explode).

