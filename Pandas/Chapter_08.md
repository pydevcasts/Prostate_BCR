# 📖 فصل ۸: ادغام و اتصال داده‌ها (Merge & Join)

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. چرا Merge & Join مهم است؟

در پروژه‌های واقعی، داده‌ها معمولاً در چندین جدول یا فایل مختلف ذخیره می‌شوند.
برای تحلیل داده‌ها باید این جداول را به هم **متصل (Join)** یا **ادغام (Merge)** کنیم.
Pandas ابزارهای قدرتمندی مثل **merge()**، **join()** و **concat()** را برای این کار فراهم می‌کند.

---

### 🔹 ۲. ادغام ساده با merge()

```python
import pandas as pd

students = pd.DataFrame({
    "ID": [1, 2, 3],
    "Name": ["Ali", "Sara", "Reza"]
})

scores = pd.DataFrame({
    "ID": [1, 2, 3],
    "Score": [85, 90, 78]
})

# Merge on column "ID"
result = pd.merge(students, scores, on="ID")
print(result)
```

📌 خروجی:

```
   ID  Name  Score
0   1   Ali     85
1   2  Sara     90
2   3  Reza     78
```

---

### 🔹 ۳. انواع اتصال‌ها (Join Types)

* **Inner Join** → فقط داده‌های مشترک بین دو جدول
* **Left Join** → همه داده‌های جدول اول + داده‌های مشترک جدول دوم
* **Right Join** → همه داده‌های جدول دوم + داده‌های مشترک جدول اول
* **Outer Join** → همه داده‌ها از هر دو جدول

```python
import pandas as pd

students = pd.DataFrame({
    "ID": [1, 2, 3, 4],
    "Name": ["Ali", "Sara", "Reza", "Omid"]
})

scores = pd.DataFrame({
    "ID": [1, 2, 5],
    "Score": [85, 90, 95]
})

print("--- Inner Join ---")
print(pd.merge(students, scores, on="ID", how="inner"))

print("\n--- Left Join ---")
print(pd.merge(students, scores, on="ID", how="left"))

print("\n--- Outer Join ---")
print(pd.merge(students, scores, on="ID", how="outer"))

```
```
--- Inner Join ---
   ID  Name  Score
0   1   Ali   85.0
1   2  Sara   90.0

--- Left Join ---
   ID  Name  Score
0   1   Ali   85.0
1   2  Sara   90.0
2   3  Reza    NaN
3   4  Omid    NaN

--- Outer Join ---
   ID  Name  Score
0   1   Ali   85.0
1   2  Sara   90.0
2   3  Reza    NaN
3   4  Omid    NaN
4   5   NaN   95.0

```
---
### 🔹 ۴. ادغام با چندین ستون

```python
data1 = pd.DataFrame({
    "FirstName": ["Ali", "Sara"],
    "LastName": ["Ahmadi", "Karimi"],
    "Age": [25, 22]
})

data2 = pd.DataFrame({
    "FirstName": ["Ali", "Sara"],
    "LastName": ["Ahmadi", "Karimi"],
    "Score": [85, 90]
})

# Merge on multiple columns
result = pd.merge(data1, data2, on=["FirstName", "LastName"])
print(result)
```

📌 خروجی:

```
  FirstName LastName  Age  Score
0       Ali   Ahmadi   25     85
1      Sara   Karimi   22     90
```

---

### 🔹 ۵. اتصال با concat()

```python
df1 = pd.DataFrame({"Name": ["Ali", "Sara"], "Score": [85, 90]})
df2 = pd.DataFrame({"Name": ["Reza", "Omid"], "Score": [78, 92]})

# Concatenate along rows
print(pd.concat([df1, df2]))

# Concatenate along columns
print(pd.concat([df1, df2], axis=1))
```

---

### 🔹 ۶. استفاده از join()

```python
df1 = pd.DataFrame({"Score": [85, 90, 78]}, index=["Ali", "Sara", "Reza"])
df2 = pd.DataFrame({"Age": [25, 22, 30]}, index=["Ali", "Sara", "Reza"])

# Join by index
print(df1.join(df2))
```

📌 خروجی:

```
      Score  Age
Ali      85   25
Sara     90   22
Reza     78   30
```

