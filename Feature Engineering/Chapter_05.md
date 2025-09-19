
# 📘 فصل پنجم: استخراج ویژگی‌ها (Feature Extraction)

🔹 در فصل قبل، در **انتخاب ویژگی‌ها (Feature Selection)** یاد گرفتیم که چگونه ویژگی‌های غیرضروری یا تکراری را شناسایی و حذف کنیم.  
اما در **استخراج ویژگی‌ها (Feature Extraction)**، ما ویژگی‌های جدیدی می‌سازیم که ترکیبی خطی (یا غیرخطی) از ویژگی‌های اصلی هستند — به شکلی که اطلاعات مهم داده‌ها را در ابعاد کمتری فشرده کنند.

---

## 🎯 چرا استخراج ویژگی مهم است؟**

- **کاهش ابعاد (Dimensionality Reduction)**: مدل‌های یادگیری ماشین در داده‌های پربُعد کند یا ناکارآمد می‌شوند (نفرین ابعاد!).
- **حذف همبستگی**: ویژگی‌های جدید (مثلاً در PCA) متعامد هستند و همبستگی بین آن‌ها صفر است.
- **حفظ اطلاعات اصلی**: با حفظ بیشترین واریانس داده، می‌توانیم بدون از دست دادن اطلاعات کلیدی، داده را فشرده کنیم.
- **تجسم بصری**: نمایش داده‌های ۳۰ بعدی در فضای ۲ یا ۳ بعدی برای درک الگوها و خوشه‌ها بسیار مفید است.

---

## 📊 ۱. استفاده از PCA (تحلیل مؤلفه‌های اصلی)**

PCA یک روش خطی برای کاهش ابعاد است که جهت‌هایی (مؤلفه‌ها) را پیدا می‌کند که بیشترین واریانس داده را در خود دارند.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# بارگذاری و پیش‌پردازش داده‌ها
X = df.drop("target", axis=1)
y = df["target"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

📌 **نکته مهم**:  
همیشه قبل از PCA، داده‌ها را **استانداردسازی** کنید — چون PCA به مقیاس ویژگی‌ها حساس است.

---

### 🔍 مرحله ۱: تحلیل واریانس تمام مؤلفه‌ها

ابتدا PCA را بدون محدودیت اجرا می‌کنیم تا ببینیم چند مؤلفه برای حفظ ۹۰٪ واریانس کافی است:

```python
# Fit PCA to find optimal number of components (90% variance threshold)
pca_full = PCA()
pca_full.fit(X_scaled)
cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_comp_90 = np.argmax(cumsum_var >= 0.90) + 1  # +1 because argmax returns index (0-based)
```
---

```python
# Fit PCA to determine optimal number of components (90% variance threshold)
pca_full = PCA()
pca_full.fit(X_scaled)

# Compute cumulative explained variance
cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)

# Find number of components needed to reach 90% explained variance
n_comp_90 = np.argmax(cumsum_var >= 0.90) + 1  # +1 because argmax returns 0-based index

# Print analysis results
print("Total number of components:", pca_full.n_components_)
print("Explained variance ratio (per component):", pca_full.explained_variance_ratio_)
print("Cumulative explained variance:", cumsum_var)
print("Number of components to retain 90% variance:", n_comp_90)
```
---

✅ **خروجی نمونه**:
```
📊 تحلیل واریانس کل مؤلفه‌ها:
تعداد کل مؤلفه‌ها: 30
نسبت واریانس توضیح‌داده‌شده (۵ مورد اول): [0.4427 0.1897 0.0939 0.0660 0.0515]
واریانس تجمعی توضیح‌داده‌شده (۵ مورد اول): [0.4427 0.6324 0.7263 0.7923 0.8438]
✅ تعداد مؤلفه‌ها برای رسیدن به 90% واریانس: 7
```
--- 

📈 **نمودار واریانس تجمعی**:

```python
# Plot cumulative explained variance to determine optimal number of components
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumsum_var) + 1), cumsum_var, marker='o', color='teal', linestyle='-', linewidth=2)
plt.axhline(y=0.90, color='red', linestyle='--', label='90% Threshold')
plt.axvline(x=n_comp_90, color='red', linestyle='--', label=f'N Components = {n_comp_90}')
plt.xlabel("Number of Components", fontsize=12)
plt.ylabel("Cumulative Explained Variance", fontsize=12)
plt.title("Explained Variance by PCA Components", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
```
---

📌 **تفسیر نمودار**:

- منحنی واریانس تجمعی نشان می‌دهد که **با تنها 7 مؤلفه، ۹۰٪ از واریانس کل داده‌ها حفظ می‌شود**.
- این یعنی می‌توانیم ابعاد داده را از **۳۰ بعد → 7 بعد** کاهش دهیم، بدون از دست دادن اطلاعات مهم.
- خط قرمز افقی و عمودی، نقطه بهینه برای انتخاب تعداد مؤلفه را نشان می‌دهد.

---

### 🎨 مرحله ۲: PCA برای تجسم دو بعدی

حالا برای نمایش بصری، PCA را با `n_components=2` اجرا می‌کنیم:

```python
# Apply PCA for 2D visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# Create DataFrame with PCA results
pca_df = pd.DataFrame(X_pca_2d, columns=["PC1", "PC2"])
pca_df["target"] = y.values
```
---

```python
# Plot PCA 2D projection with explained variance percentages
plt.figure(figsize=(10, 8))
sns.scatterplot(x="PC1", y="PC2", hue="target", data=pca_df, palette="Set1", alpha=0.7, s=60)
plt.title("PCA Projection to 2D", fontsize=14, fontweight='bold')
plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
plt.legend(title="Target", title_fontsize=12, fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```
---
‍‍‍‍‍‍‍‍
```python
# Plot cumulative explained variance to visualize PCA component efficiency
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(pca_full.explained_variance_ratio_) + 1), 
    np.cumsum(pca_full.explained_variance_ratio_), 
    marker='o', 
    color='teal',
    linestyle='-',
    linewidth=2,
    markersize=5
)
plt.xlabel("Number of Components", fontsize=12)
plt.ylabel("Cumulative Explained Variance", fontsize=12)
plt.title("Explained Variance by PCA Components", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, len(pca_full.explained_variance_ratio_) + 1, max(1, len(pca_full.explained_variance_ratio_) // 10)))
plt.tight_layout()
plt.show()
```
---

📌 **تفسیر نمودار و خروجی**:

- **PC1 تنها ۴۴.۳٪ واریانس** و **PC2 حدود ۱۹.۰٪** را توضیح می‌دهد → در مجموع **۶۳.۳٪ واریانس حفظ شده است**.
- این یعنی **حدود ۳۶.۷٪ از اطلاعات اصلی در این نمایش دو بعدی از دست رفته است** — که برای تجسم قابل قبول است، اما برای مدل‌سازی کافی نیست.
- با این حال، **کلاس‌های Benign و Malignant تا حد خوبی از هم جدا شده‌اند** — نشان‌دهنده قدرت PCA در حفظ ساختار کلاسی داده‌ها حتی در ابعاد کم.

---

🌐 **۲. روش t-SNE (برای تجسم غیرخطی و دقیق‌تر)**

t-SNE یک روش **غیرخطی** است که برای نمایش داده‌های پیچیده در فضای ۲ یا ۳ بعدی طراحی شده — و معمولاً جداسازی کلاس‌ها را بهتر از PCA نشان می‌دهد.

```python
# Apply t-SNE for non-linear dimensionality reduction and visualization
from sklearn.manifold import TSNE

# Initialize and fit t-SNE with 2 components
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# Create DataFrame for plotting
tsne_df = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2'])
tsne_df['target'] = y

# Plot t-SNE visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Dim1", y="Dim2", hue="target", data=tsne_df, palette="Set2", alpha=0.7)
plt.title("t-SNE Visualization", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

📌 **تفسیر**:

- t-SNE **خوشه‌بندی طبیعی داده‌ها** را به شکلی بسیار واضح‌تر نشان می‌دهد.
- این روش برای **تجسم و کشف الگوها** عالی است، اما **برای مدل‌سازی مناسب نیست** — چون نمی‌توان آن را روی داده جدید اعمال کرد (غیرپارامتریک است).
- پارامتر `perplexity` شبیه “تعداد همسایه‌های مؤثر” است — مقدار ۳۰ برای داده‌های متوسط مناسب است.

---

✨ **جمع‌بندی فصل**

✅ **با PCA یاد گرفتیم:**

- چگونه داده‌ها را به مؤلفه‌های جدید و متعامد تبدیل کنیم.
- چگونه تعداد بهینه مؤلفه‌ها را برای حفظ ۹۰٪ واریانس پیدا کنیم (در این مثال: ۱۰ مؤلفه).
- چگونه داده‌ها را در فضای ۲ بعدی برای تجسم نمایش دهیم — حتی با از دست دادن بخشی از اطلاعات.

✅ **با t-SNE دیدیم:**

- چگونه داده‌های پیچیده را به شکل بصری جذاب و دقیق‌تر در فضای کم‌بعدی نمایش دهیم.
- که این روش برای **درک بهتر ساختار داده** عالی است، نه برای **کاهش ابعاد عملیاتی**.

---

📌 **نکته نهایی:**

> **PCA برای مدل‌سازی و کاهش ابعاد عملیاتی مناسب است — t-SNE فقط برای تجسم!**


