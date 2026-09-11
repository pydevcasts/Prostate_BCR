#!/bin/bash
# cleanup_git_history_v2.sh
set -e

BACKUP_DIR="$HOME/git_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "📦 Backup → $BACKUP_DIR"
git bundle create "$BACKUP_DIR/repo_before_cleanup.bundle" --all
cp -r .git "$BACKUP_DIR/.git_backup" 2>/dev/null || true
echo "✅ Backup done"
echo ""

# Install filter-repo if missing
command -v git-filter-repo &> /dev/null || pip install git-filter-repo
echo "✅ git-filter-repo ready"
echo ""

# Remove ALL data directories + large notebooks
echo "🧹 Removing large files from history..."
git filter-repo --force \
  --path core/data/ \
  --path-glob '*breast-cancer-prediction-using-svm.ipynb' \
  --path-glob '*predictions-of-breast-cancer-using-svm.ipynb' \
  --path 'EnsembleLearning/Dataset' \
  --path 'Random' \
  --path-glob 'core/notebooks/02_EDA.ipynb' \
  --invert-paths

echo "✅ History cleaned"
echo ""

# Re-add remote
git remote add origin https://github.com/pydevcasts/Prostate_BCR.git 2>/dev/null || \
  git remote set-url origin https://github.com/pydevcasts/Prostate_BCR.git
echo "🔗 Remote: $(git remote get-url origin)"
echo ""

# .gitignore
cat > .gitignore << 'EOF'
# === Large data files ===
core/data/
data/
datasets/
*.csv
*.tsv
*.txt
*.gz
*.tar.gz
*.zip
*.h5
*.hdf5

# === Models ===
*.pkl
*.joblib
*.pt
*.pth
*.onnx
core/outputs/models/

# === Figures (regenerable) ===
core/outputs/figures/

# === Python ===
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# === Environment ===
.venv/
venv/
env/
.conda/

# === IDE ===
.vscode/
.idea/
*.swp
*~

# === OS ===
.DS_Store
Thumbs.db

# === Logs / backups ===
*.log
archive/
archive_*/
*_backup/

# === Secrets ===
.env
.env.*
*.key
*.pem
EOF

git add .gitignore
git commit -m "chore: add .gitignore to prevent large file commits" || true
echo "✅ .gitignore set"
echo ""

# Verify
echo "📊 .git size after cleanup:"
du -sh .git
echo ""
echo "📊 Top 10 largest remaining blobs:"
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '$1=="blob" {print $3, $4}' \
  | sort -rn \
  | head -10 \
  | awk '{printf "%.2f MB\t%s\n", $1/1024/1024, $2}'
echo ""

# Push
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "🚀 Force pushing branch: $BRANCH"
echo "⚠️  This will rewrite remote history!"
read -p "Continue? (yes/no): " CONFIRM

if [ "$CONFIRM" = "yes" ]; then
    git push origin "$BRANCH" --force
    echo "✅ Push complete"
else
    echo "❌ Push cancelled. To push later:"
    echo "   git push origin $BRANCH --force"
fi

echo ""
echo "Backup: $BACKUP_DIR"