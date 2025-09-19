
# 📖 فصل ۱۲: GroupBy و تجسم داده‌ها

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. مقدمه

یکی از قدرتمندترین قابلیت‌های **Pandas**، استفاده از متد **`groupby`** است.
با این ابزار می‌توانیم داده‌ها را براساس یک ستون گروه‌بندی کرده و آماری مانند **میانگین، مجموع، حداکثر و حداقل** بگیریم.
وقتی این نتایج را با نمودار ترکیب کنیم، درک بهتری از داده‌ها به دست می‌آوریم.

---

### 🔹 ۲. ساخت دیتافریم نمونه

```python
import pandas as pd
import matplotlib.pyplot as plt

# Sample dataset
data = pd.DataFrame({
    "Product": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
    "Month": ["Jan", "Feb", "Jan", "Feb", "Jan", "Feb", "Mar", "Mar", "Mar", "Apr"],
    "Sales": [200, 220, 150, 180, 100, 120, 250, 160, 140, 300]
})

print(data)
```

---

### 🔹 ۳. محاسبه مجموع فروش هر محصول

```python
# Group by product and sum sales
sales_sum = data.groupby("Product")["Sales"].sum()
print(sales_sum)

# Bar plot
sales_sum.plot(kind="bar", color="skyblue")
plt.title("Total Sales by Product")
plt.ylabel("Sales")
plt.show()
```

---

### 🔹 ۴. محاسبه میانگین فروش هر ماه

```python
# Group by month and average sales
monthly_avg = data.groupby("Month")["Sales"].mean()
print(monthly_avg)

# Line plot
monthly_avg.plot(kind="line", marker="o", color="green")
plt.title("Average Monthly Sales")
plt.ylabel("Sales")
plt.show()
```

---

### 🔹 ۵. چندین آماره همزمان

```python
# Multiple statistics
stats = data.groupby("Product")["Sales"].agg(["mean", "sum", "max", "min"])
print(stats)

# Boxplot to compare distributions
data.boxplot(column="Sales", by="Product")
plt.title("Sales Distribution by Product")
plt.suptitle("")  # remove default title
plt.show()
```

---

### 🔹 ۶. ترکیب GroupBy با Scatter Plot

```python
# Average sales per product-month
grouped = data.groupby(["Product", "Month"])["Sales"].mean().reset_index()

# Scatter plot
for product in grouped["Product"].unique():
    subset = grouped[grouped["Product"] == product]
    plt.scatter(subset["Month"], subset["Sales"], label=product)

plt.title("Average Sales per Product-Month")
plt.ylabel("Sales")
plt.legend()
plt.show()
```

---

### 🔹 ۷. تمرین پیشنهادی

۱. یک دیتافریم شامل فروش سه فروشگاه مختلف در ۶ ماه بسازید.
۲. مجموع فروش هر فروشگاه را محاسبه کنید و با نمودار **bar** نمایش دهید.
۳. میانگین فروش هر ماه را محاسبه کنید و با نمودار **line** نمایش دهید.
4\. توزیع فروش هر فروشگاه را با نمودار **boxplot** مقایسه کنید.

