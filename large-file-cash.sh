# 1. حذف فایل‌های بزرگ از تاریخچه Git
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch 'core/data/processed/X_features_final.csv' 'core/data/processed/X_train_preprocessed.csv' 571da6228bcfe4c4626b0aa564ba6d012fb85a23" \
  --prune-empty --tag-name-filter cat -- --all

# 2. پاکسازی
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 3. Push مجدد
git push origin main --force