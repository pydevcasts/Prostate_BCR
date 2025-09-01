
## 📖 صفحه ۱۰: بهبود مدل با انتخاب ویژگی و تنظیمات بیشتر

✍️ نویسنده: سیامک عباس‌نژاد

در صفحه قبل، مدل رگرسیون لجستیک را آموزش دادیم و نتایج اولیه نسبتاً خوب (حدود ۷۵٪ تا ۸۰٪ دقت) به دست آوردیم. اما در علم داده همیشه مرحله‌ی بعد، **بهبود مدل** است. در این صفحه با چند روش بهینه‌سازی آشنا می‌شویم.

---

### 🔹 اهمیت انتخاب ویژگی (Feature Selection)

گاهی همه‌ی ویژگی‌ها برای مدل‌سازی مفید نیستند. برخی ویژگی‌ها حتی ممکن است نویز (Noise) ایجاد کنند و دقت مدل را کاهش دهند. بنابراین انتخاب ویژگی‌های مهم می‌تواند به بهبود عملکرد کمک کند.

برای بررسی اهمیت ویژگی‌ها در رگرسیون لجستیک می‌توان از ضرایب مدل استفاده کرد:

```python
# Feature importance using model coefficients
importance = model.coef_[0]
for i, col in enumerate(X.columns):
    print(f"{col}: {importance[i]}")
```

📌 تفسیر: مقدار مثبت نشان‌دهنده‌ی افزایش احتمال ابتلا به دیابت است، و مقدار منفی اثر کاهنده دارد.

---

### 🔹 تنظیم هایپرپارامترها (Hyperparameter Tuning)

مدل رگرسیون لجستیک پارامترهایی دارد که با تغییر آن‌ها می‌توان عملکرد مدل را بهبود داد:

* **C:** معکوس شدت منظم‌سازی (Regularization). هرچه C کوچک‌تر باشد، محدودیت قوی‌تر است.
* **Penalty:** نوع نرمال‌سازی (L1 یا L2).
* **Solver:** الگوریتم بهینه‌سازی (lbfgs، saga، liblinear).

می‌توان با استفاده از جستجوی شبکه‌ای (Grid Search) بهترین مقادیر را پیدا کرد:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"]
}

grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)
```

---

### 🔹 اعتبارسنجی متقاطع (Cross Validation)

برای اطمینان از اینکه مدل ما فقط روی داده‌های آموزش خوب کار نمی‌کند (Overfitting)، از **Cross Validation** استفاده می‌کنیم.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("Cross-validation accuracy scores:", scores)
print("Mean accuracy:", scores.mean())
```

📊 این روش نشان می‌دهد که میانگین دقت مدل در چندین تقسیم مختلف داده چقدر است.

---

### 🔹 نتایج بهبود یافته

* پس از **Feature Selection** و **Tuning** دقت مدل معمولاً به حدود ۷۶٪ تا ۸۵٪ می‌رسد.
* Recall بیماران مبتلا به دیابت افزایش پیدا می‌کند که از نظر پزشکی اهمیت بالایی دارد.

---

📍 در صفحه بعد (**صفحه ۱۱**) به جمع‌بندی و نتیجه‌گیری پروژه خواهیم پرداخت و به دانشجویان نشان می‌دهیم که چگونه از این مدل ساده می‌توان در مسائل واقعی پزشکی استفاده کرد.

