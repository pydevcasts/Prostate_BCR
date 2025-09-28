
# 📖 فصل ۵: نمونه‌برداری از داده‌ها و بررسی همبستگی ویژگی‌ها

### 🔹 نمایش چند نمونه از پیام‌ها

برای درک بهتر داده‌ها، چند نمونه از پیام‌های اسپم و پیام‌های عادی (Ham) را مشاهده می‌کنیم. این کار به ما کمک می‌کند تا به صورت شهودی تفاوت محتوایی بین پیام‌های اسپم و عادی را ببینیم.

```python
# Print a few samples of spam messages
print("\nSample Spam Messages:")
print(df[df['label'] == 'spam']['message'].head(3))

# Print a few samples of ham messages
print("\nSample Ham Messages:")
print(df[df['label'] == 'ham']['message'].head(3))
```

📌 **خروجی:**

**نمونه پیام‌های اسپم (Spam):**

```
Free entry in 2 a wkly comp to win FA Cup fina...
FreeMsg Hey there darling it's been 3 week's n...
WINNER!! As a valued network customer you have...
```

**نمونه پیام‌های عادی (Ham):**

```
Go until jurong point, crazy.. Available only ...
Ok lar... Joking wif u oni...
U dun say so early hor... U c already then say...
```

📊 همانطور که دیده می‌شود، پیام‌های اسپم معمولاً شامل عباراتی مثل **WINNER!!, Free, Prize** هستند، در حالی که پیام‌های عادی بیشتر مکالمه‌های روزمره‌اند.

---

### 🔹 استخراج ویژگی‌های عددی بیشتر

برای آماده‌سازی داده‌ها جهت مدل‌سازی، ویژگی‌های عددی بیشتری ایجاد می‌کنیم. این ویژگی‌ها به مدل کمک می‌کنند تا تفاوت بین پیام‌های اسپم و عادی را بهتر یاد بگیرد.

```python
import numpy as np

# Create several numerical features from the text
df['message_length'] = df['message'].apply(len)  # Length of the message
df['word_count'] = df['message'].apply(lambda x: len(x.split()))  # Count of words in the message
df['char_count'] = df['message'].apply(lambda x: len(x.replace(" ", "")))  # Count of characters excluding spaces
df['avg_word_length'] = df['message'].apply(lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0)  # Average word length
df['exclamation_count'] = df['message'].apply(lambda x: x.count('!'))  # Count of exclamation marks
df['question_count'] = df['message'].apply(lambda x: x.count('?'))  # Count of question marks
df['digit_count'] = df['message'].apply(lambda x: sum(c.isdigit() for c in x))  # Count of digits
df['has_free'] = df['message'].str.contains('free', case=False, na=False).astype(int)  # Presence of the word 'free'
df['has_win'] = df['message'].str.contains('win', case=False, na=False).astype(int)  # Presence of the word 'win'
df['has_urgent'] = df['message'].str.contains('urgent', case=False, na=False).astype(int)  # Presence of the word 'urgent'
df['has_call'] = df['message'].str.contains('call', case=False, na=False).astype(int)  # Presence of the word 'call'

# Convert labels to numeric
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

print("Available columns after adding 'label_num':")
print(df.columns)
```

📌 در این بخش ویژگی‌هایی مثل:

* تعداد کاراکترها،
* تعداد کلمات،
* تعداد علامت‌های تعجب و سؤال،
* تعداد ارقام،
* وجود کلمات کلیدی مثل **free, win, urgent, call**

به داده‌ها اضافه شدند. این موارد نقش مهمی در تشخیص اسپم دارند.

---

### 🔹 محاسبه ماتریس همبستگی

برای بررسی ارتباط بین ویژگی‌های استخراج‌شده و برچسب‌ها، از **ماتریس همبستگی (Correlation Matrix)** استفاده می‌کنیم. این ابزار کمک می‌کند تا بفهمیم کدام ویژگی‌ها بیشترین ارتباط را با برچسب اسپم یا عادی دارند.

```python
# Select numeric columns for correlation calculation
numeric_cols = [
    'label_num',           # Numeric representation of the label (spam or ham)
    'message_length',      # Length of the message
    'word_count',          # Count of words in the message
    'char_count',          # Count of characters excluding spaces
    'avg_word_length',     # Average length of words in the message
    'exclamation_count',    # Count of exclamation marks
    'question_count',      # Count of question marks
    'digit_count',         # Count of digits
    'has_free',            # Presence of the word 'free'
    'has_win',             # Presence of the word 'win'
    'has_urgent',          # Presence of the word 'urgent'
    'has_call'             # Presence of the word 'call'
]

# Compute the correlation matrix
correlation_matrix = df[numeric_cols].corr()

print("Correlation Matrix:")
print(correlation_matrix)
```

📌 **نتیجه:** این ماتریس نشان می‌دهد که ویژگی‌هایی مانند **وجود کلمه‌ی free یا win** بیشترین همبستگی مثبت با اسپم بودن پیام دارند. در حالی که ویژگی‌هایی مثل طول پیام یا میانگین طول کلمات، تأثیر کمتری دارند.

---

✅ این بخش (صفحه ۵) به ما کمک کرد تا **ویژگی‌های مهم را شناسایی کنیم** و بفهمیم کدام یک در پیش‌بینی اسپم بودن پیام‌ها مؤثرتر هستند. در مرحله بعد می‌توانیم از این ویژگی‌ها در مدل‌های یادگیری ماشین مثل Naive Bayes استفاده کنیم.

