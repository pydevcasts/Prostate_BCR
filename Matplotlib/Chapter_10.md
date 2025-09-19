## 📖 فصل ۱۰: ترکیب Matplotlib با کتابخانه‌های دیگر

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. ترکیب با Pandas

`Pandas` قابلیت ترسیم مستقیم نمودارها را با کمک Matplotlib دارد.

```python
import pandas as pd
import matplotlib.pyplot as plt

# ساخت دیتافریم نمونه
data = {
    "Year": [2018, 2019, 2020, 2021, 2022],
    "Sales": [100, 120, 90, 150, 180]
}
df = pd.DataFrame(data)

# رسم نمودار
df.plot(x="Year", y="Sales", kind="line", marker="o", title="Sales Over Years")
plt.show()
```

📌 این روش برای تحلیل داده‌های جدولی خیلی سریع و راحت است.

---

### 🔹 ۲. ترکیب با Seaborn

`Seaborn` بر پایه Matplotlib ساخته شده و نمودارهای پیشرفته‌تر و زیباتری ارائه می‌دهد.

```python
import seaborn as sns
import numpy as np

# داده‌های نمونه
np.random.seed(42)
data = np.random.randn(100)

# هیستوگرام با Seaborn
sns.histplot(data, bins=20, kde=True, color="skyblue")
plt.title("Histogram with Seaborn")
plt.show()
```

📌 این نمودار علاوه بر هیستوگرام، منحنی توزیع احتمالی (KDE) را هم نمایش می‌دهد.

---

### 🔹 ۳. ترکیب Matplotlib و Seaborn برای شخصی‌سازی

می‌توان نمودار Seaborn را رسم کرد و سپس با دستورات Matplotlib آن را تغییر داد.

```python
tips = sns.load_dataset("tips")  # دیتاست آماده

ax = sns.boxplot(x="day", y="total_bill", data=tips, palette="Set2")
plt.title("Boxplot of Bills by Day")

# اضافه کردن خط میانگین با Matplotlib
mean_value = tips["total_bill"].mean()
plt.axhline(mean_value, color="red", linestyle="--", label="Mean")
plt.legend()
plt.show()
```

📌 این روش ترکیب قدرت Seaborn در تجسم داده‌ها با انعطاف Matplotlib برای شخصی‌سازی است.

---

### 🔹 ۴. مثال ترکیبی با Pandas + Seaborn + Matplotlib

```python
iris = sns.load_dataset("iris")

# نمودار جفتی Seaborn
sns.pairplot(iris, hue="species", diag_kind="kde")

# تغییرات اضافه با Matplotlib
plt.suptitle("Iris Dataset Pairplot", y=1.02, fontsize=14)
plt.show()
```

📌 نمودار جفتی (Pairplot) توزیع و ارتباط بین ویژگی‌ها را برای هر گونه (Species) نشان می‌دهد.

---

### 🔹 نتیجه‌گیری فصل ۱۰

* Pandas برای رسم سریع نمودارهای داده‌های جدولی عالی است.
* Seaborn ظاهر مدرن‌تر و قابلیت‌های آماری بیشتری دارد.
* ترکیب Seaborn و Matplotlib انعطاف بالایی در شخصی‌سازی می‌دهد.
