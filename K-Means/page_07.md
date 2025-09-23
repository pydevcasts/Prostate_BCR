
# 📖 مرحله پنجم: مقایسه روش بدون‌نظارت (K-Means) با روش‌های نظارت‌شده

---

## 🔹 ۱. آماده‌سازی داده‌ها برای مدل‌های نظارت‌شده

```python
from sklearn.model_selection import train_test_split

# Features and labels
X = df.iloc[:, :-2]  # همه ویژگی‌ها
y = df["target"]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
```

---

## 🔹 ۲. Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {acc_lr:.2f}")
```

📊 دقت Logistic Regression روی Iris معمولاً بین **۹۵٪ تا ۹۸٪** هست.

---

## 🔹 ۳. K-Nearest Neighbors (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {acc_knn:.2f}")
```

📊 دقت KNN معمولاً حدود **۹۵٪** هست (با انتخاب مناسب k حتی بالاتر می‌ره).

---

## 🔹 ۴. Support Vector Machine (SVM)

```python
from sklearn.svm import SVC

svm = SVC(kernel="linear")
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {acc_svm:.2f}")
```

📊 دقت SVM روی Iris معمولاً به **۹۷٪** می‌رسه.

---

## 🔹 ۵. مقایسه همه مدل‌ها با K-Means

```python
results = {
    "K-Means (Unsupervised)": acc,
    "Logistic Regression": acc_lr,
    "KNN": acc_knn,
    "SVM": acc_svm
}

# Convert to DataFrame for visualization
results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"])

# Plot
plt.figure(figsize=(7,5))
sns.barplot(x=results_df.index, y="Accuracy", data=results_df, palette="viridis")
plt.ylim(0, 1)
plt.title("Comparison of K-Means and Supervised Models on Iris Dataset")
plt.ylabel("Accuracy")
plt.show()
```

---

## 🔎 تحلیل نتایج:

* 📌 **K-Means (بدون‌نظارت):** دقت حدود **۸۹٪** → خوبه ولی به پای مدل‌های نظارت‌شده نمی‌رسه.
* 📌 **Logistic Regression:** دقت حدود **۹۶٪** → سریع و کارآمد.
* 📌 **KNN:** دقت حدود **۹۵٪** → ساده ولی نیاز به انتخاب درست k داره.
* 📌 **SVM:** بهترین عملکرد با دقت حدود **۹۷٪**.

---

✅ نتیجه‌گیری:

* وقتی برچسب‌ها نداریم (بدون‌نظارت)، K-Means انتخاب مناسبیه و می‌تونه ساختار کلی داده‌ها رو خوب پیدا کنه.
* اما وقتی برچسب‌ها موجود باشن، مدل‌های نظارت‌شده (مثل SVM یا Logistic Regression) عملکرد بهتری دارن.

---

📌 مرحله بعد (مرحله ششم) می‌تونه بررسی **روش‌های انتخاب تعداد خوشه (Elbow, Silhouette Analysis)** باشه تا بفهمیم چرا ۳ خوشه بهترین انتخاب برای Iris هست.

