# 📘 کتابچهٔ آموزشی: مدیریت داده‌های پرت در نمودار جعبه‌ای (Boxplot)

## فصل ۱: مقدمه — دادهٔ پرت چیست؟

در تحلیل داده، **دادهٔ پرت** (Outlier) به مشاهده‌ای گفته می‌شود که به‌وضوح با سایر داده‌ها همخوانی ندارد. این داده‌ها ممکن است ناشی از خطاهای اندازه‌گیری، ورود اشتباه داده، یا گاهی پدیده‌های واقعی ولی نادر باشند.

نمودار جعبه‌ای (Boxplot) یکی از ابزارهای قدرتمند برای **شناسایی بصری** داده‌های پرت است. اما گاهی نیاز داریم این داده‌ها را **از مجموعهٔ داده حذف کنیم** تا مدل‌های آماری یا یادگیری ماکین دقیق‌تر عمل کنند.

---

## فصل ۲: چگونه Boxplot دادهٔ پرت را تشخیص می‌دهد؟

در یک نمودار جعبه‌ای استاندارد، داده‌های پرت بر اساس **دامنهٔ بین‌چارکی** (IQR) شناسایی می‌شوند:

- **چارک اول (Q1)**: ۲۵امین صدک داده‌ها  
- **چارک سوم (Q3)**: ۷۵امین صدک داده‌ها  
- **IQR = Q3 − Q1**

سپس دو مرز تعریف می‌شود:

- **مرز پایینی**: `Q1 − 1.5 × IQR`  
- **مرز بالایی**: `Q3 + 1.5 × IQR`

هر داده‌ای که **کمتر از مرز پایینی** یا **بیشتر از مرز بالایی** باشد، به عنوان **دادهٔ پرت** در نظر گرفته می‌شود.

> 💡 نکته: این روش توسط جان تاکی (John Tukey) پیشنهاد شده و به‌طور گسترده در آمار کلاسیک استفاده می‌شود.

---

## فصل ۳: دو راه‌برد برای کار با داده‌های پرت

### راه‌برد ۱: **مخفی کردن داده‌های پرت در نمودار**  
فقط برای نمایش زیباتر — داده‌ها تغییری نمی‌کنند.

### راه‌برد ۲: **حذف واقعی داده‌های پرت از مجموعه داده**  
برای پاک‌سازی داده قبل از مدل‌سازی.

---

## فصل ۴: پیاده‌سازی در پایتون

در این بخش، هر دو راه‌برد را با استفاده از کتابخانه‌های `pandas` و `matplotlib` پیاده‌سازی می‌کنیم.

### ۴.۱. وارد کردن کتابخانه‌ها و تولید دادهٔ نمونه

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample data with outliers (e.g., 50 is an extreme value)
data = [1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 50]
df = pd.Series(data, name='values')
```

### ۴.۲. راه‌برد ۱: مخفی کردن داده‌های پرت در نمودار

```python
# Plot with outliers shown
plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.boxplot(df, vert=False, showfliers=True)
plt.title('Boxplot with outliers')

# Plot with outliers hidden (visual only — data unchanged)
plt.subplot(1, 2, 2)
plt.boxplot(df, vert=False, showfliers=False)  # Hide outliers in plot
plt.title('Boxplot: outliers hidden (showfliers=False)')

plt.tight_layout()
plt.show()
```

> 🔍 توضیح:  
> پارامتر `showfliers=False` فقط باعث **عدم نمایش** داده‌های پرت در نمودار می‌شود.  
> داده‌های اصلی بدون تغییر باقی می‌مانند.

### ۴.۳. راه‌برد ۲: حذف واقعی داده‌های پرت

```python
# Calculate quartiles and IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Compute lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the data: keep only values within bounds
df_filtered = df[(df >= lower_bound) & (df <= upper_bound)]

# Plot the cleaned data
plt.figure(figsize=(6, 2))
plt.boxplot(df_filtered, vert=False)
plt.title('Boxplot after removing outliers')
plt.show()

# Print results for verification
print("Original data:", df.tolist())
print("Filtered data:", df_filtered.tolist())
print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
```

### ۴.۴. تابع کاربردی برای حذف داده‌های پرت

برای استفادهٔ مکرر، می‌توانید یک تابع عمومی بنویسید:

```python
def remove_outliers_iqr(series: pd.Series) -> pd.Series:
    """
    Remove outliers from a pandas Series using the IQR method.
    
    Parameters:
        series (pd.Series): Input numerical data.
    
    Returns:
        pd.Series: Filtered series with outliers removed.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series[(series >= lower_bound) & (series <= upper_bound)]

# Usage example
clean_data = remove_outliers_iqr(df)
```

---

## فصل ۵: هشدارهای مهم

- **همیشه دلیل وجود دادهٔ پرت را بررسی کنید**. گاهی این داده‌ها ارزشمندترین بخش داده‌های شما هستند (مثلاً تشخیص کلاهبرداری).
- **حذف خودکار داده‌های پرت بدون بررسی** می‌تواند منجر به **تحلیل گمراه‌کننده** شود.
- در مسائلی مانند **مدل‌سازی رگرسیون** یا **خوشه‌بندی**، گاهی بهتر است از روش‌های **مقاوم** (Robust) استفاده کنید تا داده‌های پرت را حذف نکنید.

---

## فصل ۶: جمع‌بندی

| هدف | روش پیشنهادی |
|------|----------------|
| نمایش زیباتر نمودار | استفاده از `showfliers=False` در `matplotlib` |
| پاک‌سازی واقعی داده | فیلتر کردن داده با استفاده از مرزهای IQR |
| استفادهٔ مکرر | نوشتن تابع `remove_outliers_iqr` |

> ✅ **نکتهٔ طلایی**:  
> «دادهٔ پرت همیشه نویز نیست. گاهی صدای بلندِ واقعیت است.»

