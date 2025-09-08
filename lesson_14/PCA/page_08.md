
---

## 📖 صفحه ۸: Visualization داده‌ها در فضای جدید PCA

✍️ نویسنده: سیامک عباس‌نژاد

---

### 🔹 Visualization دوبعدی (۲D)

بعد از اجرای PCA و کاهش ابعاد داده‌ها به ۲ مؤلفه اصلی (PC1 و PC2)، می‌توانیم داده‌ها را روی یک نمودار دوبعدی نمایش دهیم.

```python
plt.figure(figsize=(8,6))
plt.scatter(X_pca_sklearn[:,0], X_pca_sklearn[:,1], c=data['target'], cmap='rainbow', edgecolor='k', s=70)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D Visualization of Wine Dataset after PCA")
plt.colorbar(label="Wine Class")
plt.show()
```

📌 تحلیل:

* همان‌طور که مشاهده می‌کنیم، کلاس‌ها به‌خوبی از هم جدا می‌شوند.
* کلاس ۰ (شراب نوع اول) فاصله زیادی از کلاس‌های دیگر دارد.
* کلاس‌های ۱ و ۲ کمی همپوشانی دارند ولی همچنان مرزبندی مشخص است.

---

### 🔹 Visualization سه‌بعدی (۳D)

گاهی اوقات استفاده از **۳ مؤلفه اصلی اول (PC1, PC2, PC3)** برای Visualization دید بهتری از توزیع داده می‌دهد.

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca_full[:,0], X_pca_full[:,1], X_pca_full[:,2], 
           c=data['target'], cmap='rainbow', edgecolor='k', s=60)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.title("3D Visualization of Wine Dataset after PCA")
plt.show()
```

📌 تحلیل:

* نمایش سه‌بعدی کمک می‌کند تا مرزبندی کلاس‌ها واضح‌تر شود.
* ترکیب PC1 و PC2 و PC3 بیش از **۶۵٪ از واریانس داده‌ها** را توضیح می‌دهد.
* در این حالت، داده‌ها با دقت بالاتری نسبت به دوبعدی جداسازی می‌شوند.

---

### 🔹 اهمیت Visualization

Visualization بعد از PCA به ما کمک می‌کند:

1. الگوهای پنهان در داده‌ها را بهتر ببینیم.
2. جداسازی کلاس‌ها را در فضای جدید بررسی کنیم.
3. بفهمیم آیا PCA توانسته داده‌ها را فشرده کند بدون اینکه ساختار اصلی از بین برود.

