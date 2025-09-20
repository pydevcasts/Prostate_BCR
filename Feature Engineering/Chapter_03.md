# 📘 فصل سوم: تحلیل تصویری ویژگی‌ها

🔹 بعد از بارگذاری داده و بررسی اولیه، وقت آن است که داده‌ها را به صورت **تصویری (Visual Analysis)** تحلیل کنیم. این کار کمک می‌کند بفهمیم:

1. توزیع ویژگی‌ها در کلاس‌های مختلف چگونه است.
2. کدام ویژگی‌ها بیشترین تمایز را بین **Benign** و **Malignant** دارند.
3. آیا ویژگی‌های ما همبستگی زیادی با هم دارند یا خیر.

---

## 📊 ۱.  برای مقایسه ویژگی‌ها(Boxplot)

Boxplot به ما نشان می‌دهد توزیع مقادیر هر ویژگی در دو کلاس (خوش‌خیم و بدخیم) چگونه است.

مثال روی ویژگی **mean radius**:

```python
plt.figure(figsize=(8,5))
sns.boxplot(x="target", y="mean radius", data=df, palette="Set1")
plt.xticks([0, 1], ['Malignant', 'Benign'])
plt.title("Boxplot of Mean Radius by Target")
plt.show()
```

📌 تفسیر:

* بیماران بدخیم (Malignant) معمولاً میانگین شعاع سلول‌های بزرگ‌تری دارند.
* این ویژگی یکی از شاخص‌های مهم در تشخیص سرطان است.

---

## 🔥 ۲. Heatmap برای همبستگی ویژگی‌ها

با استفاده از Heatmap می‌توانیم بررسی کنیم کدام ویژگی‌ها همبستگی زیادی با هم دارند.

```python
plt.figure(figsize=(12,10))
corr = df.drop("target", axis=1).corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap of Features")
plt.show()
```

📌 تفسیر:

* برخی ویژگی‌ها مثل **mean radius**، **mean perimeter** و **mean area** همبستگی بسیار بالایی دارند.
* این یعنی احتمالاً همه‌ی این ویژگی‌ها اطلاعات مشابهی را منتقل می‌کنند و شاید نیازی نباشد همه‌ی آن‌ها را نگه داریم.

---

## 🎻 ۳. Violinplot برای توزیع ویژگی‌ها

Violinplot ترکیبی از Boxplot و Kernel Density است.

مثال روی ویژگی **mean concavity**:

```python
plt.figure(figsize=(8,5))
sns.violinplot(x="target", y="mean concavity", data=df, palette="Set2")
plt.xticks([0, 1], ['Malignant', 'Benign'])
plt.title("Violinplot of Mean Concavity by Target")
plt.show()
```

📌 تفسیر:

* بیماران بدخیم مقادیر بیشتری در concavity دارند.
* این ویژگی برای جداسازی کلاس‌ها مهم به نظر می‌رسد.

---

## 🔗 ۴. Pairplot برای نمایش چند ویژگی همزمان

Pairplot به ما امکان می‌دهد روابط بین چند ویژگی را در دو کلاس مختلف مشاهده کنیم.

```python
sns.pairplot(df[['mean radius', 'mean texture', 'mean concavity', 'target']], 
             hue="target", palette="Set1")
plt.show()
```

📌 تفسیر:

* ترکیب ویژگی‌ها مثل **mean radius** و **mean concavity** می‌تواند مرز مشخصی بین بدخیم و خوش‌خیم ایجاد کند.
* برخی ویژگی‌ها خیلی خوب کلاس‌ها را جدا می‌کنند.

---

## ✨ جمع‌بندی فصل

در این فصل یاد گرفتیم:

1. Boxplot نشان داد که برخی ویژگی‌ها مثل **mean radius** در بیماران بدخیم مقادیر بالاتری دارند.
2. Heatmap آشکار کرد که بعضی ویژگی‌ها همبستگی بالایی دارند و احتمالاً باید یکی از آن‌ها حذف شود.
3. Violinplot کمک کرد توزیع دقیق ویژگی‌ها را در دو کلاس ببینیم.
4. Pairplot نشان داد ترکیب چند ویژگی برای جداسازی کلاس‌ها خیلی مفید است.

