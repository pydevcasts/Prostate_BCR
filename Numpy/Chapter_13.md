# 📖 فصل ۱۳: کار با داده‌های تصادفی در NumPy

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. تولید اعداد تصادفی ساده

```python
import numpy as np

print("Random between 0 and 1:", np.random.rand())     # عدد بین 0 و 1
print("Array 1x5 random:", np.random.rand(5))          # آرایه یک‌بعدی
print("Matrix 2x3 random:\n", np.random.rand(2, 3))    # ماتریس دوبعدی
```

📌 تابع `rand` اعداد **یونیفرم بین ۰ و ۱** تولید می‌کند.

---

### 🔹 ۲. اعداد صحیح تصادفی

```python
arr = np.random.randint(1, 10, size=(3, 3))
print("Random integers:\n", arr)
```

📌 با `randint` می‌توان بازه اعداد و شکل آرایه را مشخص کرد.

---

### 🔹 ۳. توزیع نرمال (Normal Distribution)

```python
data = np.random.normal(loc=50, scale=10, size=1000)  # میانگین=50، انحراف معیار=10
print("Mean:", np.mean(data))
print("Std:", np.std(data))
```

📌 این تابع بسیار کاربردی است؛ چون در بسیاری از داده‌های واقعی توزیع نرمال دیده می‌شود.

---

### 🔹 ۴. توزیع‌های آماری دیگر

```python
print("Uniform:", np.random.uniform(0, 1, 5))   # یکنواخت
print("Binomial:", np.random.binomial(10, 0.5, 5)) # دوجمله‌ای
print("Poisson:", np.random.poisson(3, 5))  # پواسون
```

📌 NumPy مجموعه‌ای از توزیع‌های مختلف دارد که در آمار و شبیه‌سازی استفاده می‌شود.

---

### 🔹 ۵. انتخاب تصادفی از یک لیست

```python
names = ["Ali", "Sara", "Reza", "Niloofar"]

choice = np.random.choice(names, size=3, replace=False)
print("Random choice:", choice)
```

📌 اگر `replace=False` باشد انتخاب بدون تکرار انجام می‌شود.

---

### 🔹 ۶. بازتولید نتایج با Seed

```python
np.random.seed(42)
print(np.random.rand(3))
```

📌 با **seed** می‌توانیم تولید اعداد تصادفی را تکرارپذیر کنیم.

---

### 🔹 ۷. مثال تصویری با Matplotlib 🎨

```python
import matplotlib.pyplot as plt

data = np.random.normal(0, 1, 1000)

plt.hist(data, bins=30, alpha=0.7, color="purple", edgecolor="black")
plt.title("Normal Distribution (Mean=0, Std=1)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

📌 این نمودار هیستوگرام داده‌های نرمال را نمایش می‌دهد.

---

### 🔹 ۸. تمرین پیشنهادی 🎯

۱. یک ماتریس ۵×۵ از اعداد تصادفی بین ۱۰ تا ۱۰۰ تولید کنید.
۲. یک آرایه ۱۰۰۰ تایی با توزیع نرمال (میانگین ۱۰۰، انحراف معیار ۱۵) بسازید و هیستوگرام آن را رسم کنید.
۳. از لیست نام‌ها، ۵ نفر را به‌صورت تصادفی انتخاب کنید.
۴. یک توزیع دوجمله‌ای شبیه‌سازی کنید و نمودارش را رسم کنید.

