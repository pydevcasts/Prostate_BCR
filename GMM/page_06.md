## 🔹 مصورسازی مسئولیت‌ها با Barplot و Heatmap

```python
import seaborn as sns

# مسئولیت‌ها از GMM
responsibilities = gmm.predict_proba(X.reshape(-1, 1))

# ---------- Barplot ----------
plt.figure(figsize=(8, 5))
for i in range(len(X)):
    plt.bar(i, responsibilities[i, 0], color="blue", alpha=0.6)
    plt.bar(i, responsibilities[i, 1], bottom=responsibilities[i, 0], color="orange", alpha=0.6)

plt.xticks(range(len(X)), X)
plt.xlabel("Data Points")
plt.ylabel("Responsibilities (γ)")
plt.title("Responsibilities of Each Data Point to Two Components")
plt.legend(["Component 1", "Component 2"])
plt.show()

# ---------- Heatmap ----------
plt.figure(figsize=(6, 4))
sns.heatmap(responsibilities, annot=True, cmap="coolwarm", xticklabels=["Comp 1", "Comp 2"], yticklabels=X)
plt.title("Heatmap of Responsibilities (γ_ik)")
plt.xlabel("Components")
plt.ylabel("Data Points (x)")
plt.show()
```

---

## 🔹 توضیح تحلیل نتایج

1. **Barplot**:

   * هر ستون مربوط به یک داده است.
   * رنگ آبی = احتمال تعلق به خوشه‌ی اول.
   * رنگ نارنجی = احتمال تعلق به خوشه‌ی دوم.
   * مثلاً داده‌ی ۲ تقریباً احتمال بالایی در خوشه‌ی اول دارد، ولی داده‌ی ۹ بیشتر در خوشه‌ی دوم.

2. **Heatmap**:

   * یک ماتریس احتمال نشان داده می‌شود.
   * هر سطر یک داده و هر ستون یک خوشه است.
   * اعداد داخل خانه‌ها همان γ (مسئولیت‌ها) هستند.
   * هر چه رنگ به سمت قرمز تیره یا آبی پررنگ برود، احتمال تعلق بیشتر است.

---

📘 این ترکیب باعث می‌شه دانشجو دقیقاً درک کنه که **GMM فقط برچسب قطعی خوشه‌ها رو نمی‌ده، بلکه یک "وزن احتمال" برای هر داده نسبت به هر خوشه برمی‌گردونه**؛ چیزی که قدرت اصلی این الگوریتم نسبت به k-Means هست.

