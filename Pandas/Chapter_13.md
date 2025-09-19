# 📖 فصل ۱۳: Pivot Table و Crosstab در Pandas

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. مقدمه

گاهی نیاز داریم داده‌ها رو مثل **Pivot Table در Excel** خلاصه کنیم.
Pandas ابزارهای قدرتمندی مثل:

* **`pivot_table`** → برای خلاصه‌سازی انعطاف‌پذیر داده‌ها
* **`crosstab`** → برای شمارش و مقایسه مقادیر بین دسته‌ها

رو در اختیار ما قرار می‌ده.

---

### 🔹 ۲. ساخت دیتافریم نمونه

```python
import pandas as pd

# Sample dataset
data = pd.DataFrame({
    "Product": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
    "Region": ["North", "South", "North", "South", "North", "South", "North", "South", "North", "South"],
    "Sales": [200, 220, 150, 180, 100, 120, 250, 160, 140, 300]
})

print(data)
```

---

### 🔹 ۳. Pivot Table ساده

```python
# Create pivot table (average sales per product and region)
pivot = pd.pivot_table(data, values="Sales", index="Product", columns="Region", aggfunc="mean")
print(pivot)
```

📌 خروجی:

```
Region   North   South
Product                
A        225.0   260.0
B        150.0   170.0
C        120.0   120.0
```

---

### 🔹 ۴. Pivot Table با چند تابع آماری

```python
# Multiple aggregations
pivot2 = pd.pivot_table(
    data,
    values="Sales",
    index="Product",
    columns="Region",
    aggfunc=["mean", "sum", "max"]
)
print(pivot2)
```

---

### 🔹 ۵. Crosstab برای شمارش

```python
# Count number of records for each product-region
cross = pd.crosstab(data["Product"], data["Region"])
print(cross)
```

📌 خروجی:

```
Region   North  South
Product               
A            2      2
B            1      2
C            2      1
```

---

### 🔹 ۶. Crosstab با مقادیر و نرمال‌سازی

```python
# Normalize counts to get proportions
cross_norm = pd.crosstab(data["Product"], data["Region"], normalize="index")
print(cross_norm)
```

📌 خروجی:

```
Region      North  South
Product                    
A           0.50   0.50
B           0.33   0.67
C           0.67   0.33
```

---

### 🔹 ۷. تجسم Pivot Table

```python
import matplotlib.pyplot as plt

pivot.plot(kind="bar")
plt.title("Average Sales by Product and Region")
plt.ylabel("Sales")
plt.show()
```

---

### 🔹 ۸. تمرین پیشنهادی

۱. یک دیتافریم شامل فروش چند فروشگاه در سه فصل مختلف بسازید.
۲. میانگین فروش هر فروشگاه در هر فصل رو با **pivot\_table** محاسبه کنید.
3\. تعداد کل رکوردهای هر فروشگاه در هر فصل رو با **crosstab** محاسبه کنید.
4\. نتایج Pivot Table رو با نمودار **bar** رسم کنید.
