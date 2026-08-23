#!/bin/bash

echo -e "\033[1;33mحذف فایل‌های بزرگ مشخص شده...\033[0m"

# لیست فایل‌های بزرگ شما
FILES=(
    "core/data/processed/X_test_preprocessed.csv"
    "core/data/processed/X_train_preprocessed.csv"
    "core/data/raw/data_mrna_seq_v2_rsem.txt"
    "core/data/processed/X_features_final.csv"
)

# حذف هر فایل
for FILE in "${FILES[@]}"; do
    echo -e "\033[0;32mحذف: $FILE\033[0m"
    git filter-repo --path "$FILE" --invert-paths --force
done

# پاکسازی نهایی
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo -e "\033[0;32m✅ پاکسازی کامل شد!\033[0m"
echo -e "حجم جدید: $(du -sh .git | awk '{print $1}')"