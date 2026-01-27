
## **نمونه‌سازی با SMOTE و آموزش Voting Ensemble — کد و تحلیل**

✍️ *نویسنده: سیامک عباس‌نژاد*

### ۱ — آماده‌سازی X و Y و مقیاس‌بندی

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Separating features (X) and target variable (Y)
X = df_fe.drop(['Exited'], axis=1)  # Drop the 'Exited' column to create feature set
Y = df_fe['Exited']  # 'Exited' column as the target variable

# Feature scaling
sc = StandardScaler()  # Create a StandardScaler object
X_scaled = sc.fit_transform(X)  # Fit the scaler and transform the feature set
X = pd.DataFrame(X_scaled, columns=X.columns)  # Convert the scaled features back to DataFrame

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

# Print the shapes of the training and testing sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

### ۲ — نمونه‌سازی با SMOTEN (جزء کد تو) — متوازن‌سازی

```python
from collections import Counter
from imblearn.over_sampling import SMOTEN

sm = SMOTEN(random_state=0)
X_smot_train, Y_smot_train = sm.fit_resample(X_train, y_train)
X_smot_test, Y_smot_test = sm.fit_resample(X_test, y_test)

print('Original dataset shape(Train): ', Counter(y_train))
print('Resampled dataset shape(Train): ', Counter(Y_smot_train))
print('\nOriginal dataset shape(Test): ', Counter(y_test))
print('Resampled dataset shape(Test): ', Counter(Y_smot_test))
```

**یادداشت مهم:**

> خوب SMOTEN برای ویژگی‌های گسستهِ کم‌بعد کاربردی است؛ اگر فیچرهای پیوسته و زیاد داری، معمولاً `SMOTE` یا `SMOTEENN` هم کاربردی‌اند.

> تو نسخهٔ تو، SMOTEN روی همۀ فیچرها اجرا شده — اگر برخی فیچرها کاملاً باینری/کاتگوریکال‌اند ممکن است نیازمند تنظیم باشه.

---

### ۳ — تعریف مدل‌ها و VotingClassifier (کد اجراپذیر)

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import xgboost as xgb

# Define the models
rf_model = RandomForestClassifier(n_estimators=90, random_state=0)  # Random Forest model
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=0)  # Gradient Boosting model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='auc', n_estimators=100, random_state=0)  # XGBoost model

# Create a Voting Classifier with soft voting
voting_model = VotingClassifier(estimators=[
    ('rf', rf_model),
    ('gb', gb_model),
    ('xgb', xgb_model)
], voting='soft', n_jobs=-1)  # n_jobs=-1 uses all available cores

# Fit the Voting Classifier on the SMOTE-augmented training data
voting_model.fit(X_smot_train, Y_smot_train)

# Make predictions on the SMOTE-augmented test data
y_pred = voting_model.predict(X_smot_test)
```

---

### ۴ — ارزیابی مدل و متریک‌ها (کد + خروجی نمونه)

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Calculate metrics
accuracy = accuracy_score(Y_smot_test, y_pred)  # Calculate accuracy
conf_matrix = confusion_matrix(Y_smot_test, y_pred)  # Generate confusion matrix
class_report = classification_report(Y_smot_test, y_pred)  # Generate classification report
y_pred_proba = voting_model.predict_proba(X_smot_test)[:, 1]  # Get predicted probabilities for the positive class
auc = roc_auc_score(Y_smot_test, y_pred_proba)  # Calculate AUC

# Print metrics
print(f'Accuracy: {accuracy:.2f}')  # Print accuracy
print('Confusion Matrix:')
print(conf_matrix)  # Print confusion matrix
print('Classification Report:')
print(class_report)  # Print classification report
print(f'AUC: {auc:.3f}')  # Print AUC
```

**خروجی نمونه (بر اساس اجرای تو):**

```
Accuracy: 0.92
Confusion Matrix:
[[1528   57]
 [ 212 1373]]
Classification Report: (تعاریف precision/recall/f1 برای هر کلاس)
AUC: ~0.94
```

**تفسیر:**

* دقت کلی ۰.۹۲ نشان‌دهندهٔ عملکرد بسیار خوب مدل ترکیبی است.
* ماتریس نشان می‌دهد که خطاهای نوع اول (False Positives) و نوع دوم (False Negatives) در سطح متعادل قرار دارند.
* AUC ≈ 0.94 نشانگر توانایی بالا در تفکیک دو کلاس است.

---

### ۵ — رسم Confusion Matrix و ROC (کد)

```python
# Confusion matrix heatmap
plt.figure(figsize=(7,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Exited','Exited'], yticklabels=['Not Exited','Exited'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(Y_smot_test, y_pred_proba)
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0,1],[0,1], linestyle='--', color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```
