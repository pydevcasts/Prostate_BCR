# 📖 فصل ۱۴: داده‌های چندسطحی (MultiIndex) در Pandas

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. مقدمه

گاهی داده‌ها پیچیده‌تر از یک جدول ساده هستند.
مثلاً ممکنه بخوایم:

* داده‌ها رو همزمان بر اساس **محصول** و **منطقه** ذخیره کنیم
* یا برای هر سال چندین **زیرستون** داشته باشیم

اینجاست که **MultiIndex** (اندیس چندسطحی) به کمک ما میاد.

---

### 🔹 ۲. ساخت MultiIndex به صورت دستی

```python
import pandas as pd

# Create MultiIndex manually
index = pd.MultiIndex.from_tuples(
    [("A", "North"), ("A", "South"), ("B", "North"), ("B", "South"), ("C", "North")],
    names=["Product", "Region"]
)

data = pd.DataFrame({"Sales": [200, 220, 150, 180, 100]}, index=index)
print(data)
```

📌 خروجی:

```
              Sales
Product Region      
A       North    200
        South    220
B       North    150
        South    180
C       North    100
```

---

### 🔹 ۳. انتخاب داده‌ها در MultiIndex

```python
# Select all sales for product A
print(data.loc["A"])

# Select sales for product A in South
print(data.loc[("A", "South")])
```

---

### 🔹 ۴. استفاده از ستون چندسطحی

```python
# Create DataFrame with MultiIndex columns
df = pd.DataFrame(
    {
        ("2022", "Q1"): [100, 200, 300],
        ("2022", "Q2"): [150, 250, 350],
        ("2023", "Q1"): [120, 220, 320],
        ("2023", "Q2"): [180, 280, 380]
    },
    index=["Store1", "Store2", "Store3"]
)

print(df)
```

📌 خروجی:

```
        2022       2023     
          Q1   Q2    Q1   Q2
Store1   100  150   120  180
Store2   200  250   220  280
Store3   300  350   320  380
```

---

### 🔹 ۵. انتخاب داده‌ها از ستون چندسطحی

```python
# Select all Q1 data
print(df["2022"]["Q1"])

# Select Store1 Q2 of 2023
print(df.loc["Store1", ("2023", "Q2")])
```

---

### 🔹 ۶. تبدیل داده‌های چندسطحی

```python
# Stack columns into rows
stacked = df.stack()
print(stacked)

# Unstack rows into columns
unstacked = data.unstack()
print(unstacked)
```

---

### 🔹 ۷. تمرین پیشنهادی

۱. یک دیتافریم با فروش سه محصول در دو منطقه (شمال و جنوب) طی دو سال بسازید.
۲. داده‌ها رو با **MultiIndex روی ردیف‌ها** ذخیره کنید.
3\. یک **Pivot Table** بسازید و سپس نتیجه رو به **MultiIndex روی ستون‌ها** تبدیل کنید.
4\. داده‌ها رو یک بار **stack** و یک بار **unstack** کنید و نتایج رو مقایسه کنید.

