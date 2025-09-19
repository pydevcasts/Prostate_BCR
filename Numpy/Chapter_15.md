# 📖 فصل ۱۵: پروژه نهایی – تحلیل داده‌ها با NumPy و Matplotlib

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. معرفی پروژه

در این پروژه قصد داریم با استفاده از **NumPy** داده‌های مصنوعی ایجاد کنیم، سپس آن‌ها را با **Matplotlib** تحلیل و مصورسازی کنیم.
سناریو:

* شبیه‌سازی نمرات دانشجویان در ۳ درس (ریاضی، فیزیک، برنامه‌نویسی).
* محاسبه شاخص‌های آماری (میانگین، میانه، انحراف معیار).
* ترسیم نمودارهای مختلف برای تحلیل داده‌ها.

---

### 🔹 ۲. تولید داده‌ها با NumPy

```python
import numpy as np

# Set the random seed for reproducibility
np.random.seed(42)

# Generate scores for 100 students using a normal distribution

# Math scores are generated with a mean of 70 and a standard deviation of 10
math_scores = np.random.normal(70, 10, 100)        # Math

# Physics scores are generated with a mean of 65 and a standard deviation of 12
physics_scores = np.random.normal(65, 12, 100)     # Physics

# Programming scores are generated with a mean of 75 and a standard deviation of 8
programming_scores = np.random.normal(75, 8, 100)  # Programming
```

📌 میانگین و انحراف معیار نمرات هر درس متفاوت است.

---

### 🔹 ۳. محاسبات آماری

```python
print("Math - Mean:", np.mean(math_scores), " Std:", np.std(math_scores))
print("Physics - Mean:", np.mean(physics_scores), " Std:", np.std(physics_scores))
print("Programming - Mean:", np.mean(programming_scores), " Std:", np.std(programming_scores))
```

📌 به این ترتیب می‌توانیم وضعیت کلی هر درس را بررسی کنیم.

---

### 🔹 ۴. ترسیم هیستوگرام نمرات

```python
import matplotlib.pyplot as plt

plt.hist(math_scores, bins=15, alpha=0.5, label="Math", color="blue")
plt.hist(physics_scores, bins=15, alpha=0.5, label="Physics", color="green")
plt.hist(programming_scores, bins=15, alpha=0.5, label="Programming", color="orange")

plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()
```

📌 هیستوگرام‌ها نشان می‌دهند نمرات هر درس حول چه مقداری متمرکز هستند.

---

### 🔹 ۵. نمودار جعبه‌ای (Boxplot) برای مقایسه

```python
data = [math_scores, physics_scores, programming_scores]

plt.boxplot(data, labels=["Math", "Physics", "Programming"], patch_artist=True)
plt.title("Boxplot of Student Scores")
plt.ylabel("Score")
plt.show()
```

📌 Boxplot به خوبی میانه، چارک‌ها و داده‌های پرت (Outliers) را نمایش می‌دهد.

---

### 🔹 ۶. مقایسه میانگین‌ها با Bar Plot

```python
means = [np.mean(math_scores), np.mean(physics_scores), np.mean(programming_scores)]

plt.bar(["Math", "Physics", "Programming"], means, color=["blue", "green", "orange"])
plt.title("Average Scores by Subject")
plt.ylabel("Mean Score")
plt.show()
```

📌 این نمودار میانگین‌ها را کنار هم نشان می‌دهد.

---

### 🔹 ۷. نمودار Scatter برای همبستگی

```python
plt.scatter(math_scores, physics_scores, alpha=0.7, color="purple")
plt.title("Math vs Physics Scores")
plt.xlabel("Math")
plt.ylabel("Physics")
plt.show()
```

📌 با این نمودار می‌توان فهمید آیا بین دو درس همبستگی وجود دارد یا خیر.

---

### 🔹 ۸. تمرین نهایی 🎯

۱. میانگین و انحراف معیار سه درس را در یک جدول نشان دهید.
۲. نمودار خطی (Line Plot) از میانگین نمرات هر درس رسم کنید.
۳. یک scatter دیگر بین **ریاضی و برنامه‌نویسی** رسم کنید.
۴. یک تحلیل متنی کوتاه از نتایج به زبان خودتان بنویسید.

