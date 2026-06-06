# 📖 فصل ۶: انتخاب و فیلتر کردن داده‌ها (Selection & Filtering)

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 ۱. انتخاب ستون‌ها از DataFrame

گاهی فقط به یک یا چند ستون نیاز داریم:

```python
import pandas as pd

data = {
    "Name": ["Ali", "Sara", "Reza", "Niloofar"],
    "Age": [25, 22, 30, 28],
    "Score": [85, 90, 78, 95]
}

df = pd.DataFrame(data)

# Select a single column
print(df["Name"])

# Select multiple columns
print(df[["Name", "Score"]])
```

---

### 🔹 ۲. انتخاب ردیف‌ها با loc و iloc

```python
# Select row by label
print(df.loc[0])

# Select row by index number
print(df.iloc[1])
```

---

### 🔹 ۳. برش (Slicing) روی ردیف‌ها

```python
# Select first 2 rows
print(df[:2])

# Select rows from index 1 to 3
print(df[1:3])
```

---

### 🔹 ۴. فیلتر کردن بر اساس شرط‌ها

```python
# Students with Score > 80
print(df[df["Score"] > 80])

# Students Age < 28
print(df[df["Age"] < 28])
```

---

### 🔹 ۵. ترکیب شرط‌ها

```python
# Students with Age > 25 AND Score > 80
print(df[(df["Age"] > 25) & (df["Score"] > 80)])

# Students with Age < 25 OR Score < 85
print(df[(df["Age"] < 25) | (df["Score"] < 85)])
```

---

### 🔹 ۶. مرتب‌سازی داده‌ها

```python
# Sort by Age ascending
print(df.sort_values("Age"))

# Sort by Score descending
print(df.sort_values("Score", ascending=False))
```

---

### 🔹 ۷. انتخاب مقادیر خاص با isin

```python
# Select students with specific names
print(df[df["Name"].isin(["Ali", "Reza"])])
```

### 🔹 ۸ پیدا کردن نام دانش اموزی که بیشترین نمره را گرفته

```python
df.loc[df["Score"].idxmax(), "Name"]
```