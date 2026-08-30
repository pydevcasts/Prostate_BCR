# ۱. حذف همه فایل‌های CSV و TXT و GZ بزرگ
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch \
    core/data/processed/X_train_preprocessed.csv \
    core/data/processed/X_features_final.csv \
    core/data/processed/X_test_preprocessed.csv \
    core/data/raw/data_mrna_seq_v2_rsem.txt \
    core/data/external/GSE70769_family.soft.gz" \
  --prune-empty --tag-name-filter cat -- --all

# ۲. پاکسازی
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# ۳. اضافه کردن ریموت (اگر پاک شده)
git remote add origin https://github.com/pydevcasts/Prostate_BCR.git

# ۴. Push با --force
# git push origin pso-optimized-gene-signatures-for-prostate-cancer-prediction-3cd90 --force
# # ۱. نصب
# pip install git-filter-repo

# # ۲. حذف فایل‌های بزرگ
# git filter-repo --strip-blobs-bigger-than 50M

# # ۳. اضافه کردن ریموت
# git remote add origin https://github.com/pydevcasts/Prostate_BCR.git

# # ۴. Push
# git push origin pso-optimized-gene-signatures-for-prostate-cancer-prediction-3cd90 --force
# ۱. با filter-repo همه فایل‌های outputs را حذف کنید
# git filter-repo --path core/outputs/ --invert-paths --force

# # ۲. ریموت را دوباره اضافه کنید
# git remote add origin https://github.com/pydevcasts/Prostate_BCR.git

# # ۳. push کنید
# git push origin prostate-cancer-bcr-prediction-enhancement-8b910 --force