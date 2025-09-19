# 📖 فصل ۷: گروه‌بندی داده‌ها (GroupBy)

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. معرفی GroupBy

گاهی نیاز داریم داده‌ها را بر اساس یک ستون **گروه‌بندی** کنیم و روی هر گروه عملیات آماری انجام دهیم.
مثلاً: میانگین نمرات هر کلاس، تعداد افراد هر گروه سنی، بیشترین مقدار فروش هر فروشنده و …

---

### 🔹 ۲. گروه‌بندی ساده

```python
import pandas as pd

data = {
    "Class": ["A", "A", "B", "B", "C", "C"],
    "Name": ["Ali", "Sara", "Reza", "Niloofar", "Omid", "Maryam"],
    "Score": [85, 90, 78, 95, 88, 92]
}

df = pd.DataFrame(data)

# Group by Class and calculate mean
print(df.groupby("Class")["Score"].mean())
```

📌 خروجی:

```
Class
A    87.5
B    86.5
C    90.0
Name: Score, dtype: float64
```

---

### 🔹 ۳. چندین محاسبه روی گروه‌ها

```python
# Multiple aggregations
print(df.groupby("Class")["Score"].agg(["mean", "max", "min"]))
```

📌 خروجی:

```
       mean  max  min
Class                  
A       87.5   90   85
B       86.5   95   78
C       90.0   92   88
```

---

### 🔹 ۴. گروه‌بندی با چند ستون

```python
data = {
    "Department": ["IT", "IT", "HR", "HR", "Sales", "Sales"],
    "Gender": ["M", "F", "M", "F", "M", "F"],
    "Salary": [5000, 4800, 4200, 4000, 5500, 5300]
}

df = pd.DataFrame(data)

# Group by Department and Gender
print(df.groupby(["Department", "Gender"])["Salary"].mean())
```

📌 خروجی:

```
Department  Gender
HR          F         4000
            M         4200
IT          F         4800
            M         5000
Sales       F         5300
            M         5500
Name: Salary, dtype: int64
```

---

### 🔹 ۵. استفاده از size و count

```python
# Count number of records in each group
print(df.groupby("Department").size())

# Count non-null values in each group
print(df.groupby("Department")["Salary"].count())
```

---

### 🔹 ۶. مرتب‌سازی نتایج گروه‌بندی

```python
# Sort by mean Salary
grouped = df.groupby("Department")["Salary"].mean()
print(grouped.sort_values(ascending=False))
```
