#!/bin/bash
# =====================================================================
# clean_big_files.sh
# Dynamically find and remove large files from git history.
# - Auto-detects large files (default threshold: 5 MB)
# - Shows a preview before acting
# - Requires explicit confirmation
# - Creates a backup before rewriting history
# - Uses git-filter-repo (official tool, safe and fast)
# =====================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
SIZE_THRESHOLD_MB="${SIZE_THRESHOLD_MB:-5}"     # default 5 MB
BACKUP_ENABLED="${BACKUP_ENABLED:-yes}"         # yes / no
AUTO_CONFIRM="${AUTO_CONFIRM:-no}"              # yes / no
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
REMOTE_URL="${REMOTE_URL:-}"                    # optional, if empty → read from git

# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
red()   { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }
blue()  { printf "\033[34m%s\033[0m\n" "$*"; }

require_clean_tree() {
  if [[ -n "$(git status --porcelain)" ]]; then
    yellow "⚠️  Working tree is not clean."
    echo "   Commit or stash your changes first, then re-run."
    echo "   To force, set AUTO_CONFIRM=yes."
    if [[ "$AUTO_CONFIRM" != "yes" ]]; then
      exit 1
    fi
  fi
}

require_git() {
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    red "❌ Not inside a git repository."
    exit 1
  fi
}

install_filter_repo() {
  if ! command -v git-filter-repo >/dev/null 2>&1; then
    blue "📥 Installing git-filter-repo..."
    pip install --quiet git-filter-repo || {
      red "❌ Failed to install git-filter-repo."
      echo "   Try: pip install --user git-filter-repo"
      exit 1
    }
  fi
  green "✅ git-filter-repo available."
}

# ─────────────────────────────────────────────────────────────────────
# STEP 1 — Find large files in current HEAD tree
# ─────────────────────────────────────────────────────────────────────
find_large_files_in_head() {
  local threshold_bytes=$(( SIZE_THRESHOLD_MB * 1024 * 1024 ))
  git ls-tree -r -l HEAD \
    | awk -v t="$threshold_bytes" '$4+0 >= t {print $4, $5}' \
    | sort -rn \
    | awk '{printf "%.2f MB\t%s\n", $1/1024/1024, $2}'
}

# ─────────────────────────────────────────────────────────────────────
# STEP 2 — Find large blobs anywhere in history (even if deleted)
# ─────────────────────────────────────────────────────────────────────
find_large_blobs_anywhere() {
  local threshold_bytes=$(( SIZE_THRESHOLD_MB * 1024 * 1024 ))
  git rev-list --objects --all \
    | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
    | awk -v t="$threshold_bytes" '$1=="blob" && $3+0 >= t {print $3, $4}' \
    | sort -rn \
    | awk '!seen[$2]++ {printf "%.2f MB\t%s\n", $1/1024/1024, $2}'
}

# ─────────────────────────────────────────────────────────────────────
# STEP 3 — Build --path args for filter-repo from a file list
# ─────────────────────────────────────────────────────────────────────
build_filter_args() {
  local list_file="$1"
  local args=()
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    args+=( --path "$line" )
  done < "$list_file"
  printf '%s\n' "${args[@]}"
}

# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
main() {
  require_git
  require_clean_tree
  install_filter_repo

  yellow "════════════════════════════════════════════════════════════════"
  yellow "  GIT HISTORY CLEANUP — Dynamic Large-File Remover"
  yellow "════════════════════════════════════════════════════════════════"
  echo "Repository : $(pwd)"
  echo "Branch     : $BRANCH"
  echo "Threshold  : ${SIZE_THRESHOLD_MB} MB"
  echo ""

  # ── List large files in HEAD ──────────────────────────────────────
  blue "🔍 Large files in current HEAD (>= ${SIZE_THRESHOLD_MB} MB):"
  find_large_files_in_head | tee /tmp/large_head.txt
  echo ""

  # ── List large blobs anywhere in history ──────────────────────────
  blue "🔍 Large blobs anywhere in history (>= ${SIZE_THRESHOLD_MB} MB):"
  find_large_blobs_anywhere | tee /tmp/large_history.txt
  echo ""

  # ── Collect unique paths to remove ────────────────────────────────
  awk '{print $2}' /tmp/large_history.txt | sort -u > /tmp/paths_to_remove.txt
  local count
  count=$(wc -l < /tmp/paths_to_remove.txt | tr -d ' ')
  if [[ "$count" -eq 0 ]]; then
    green "✅ No files >= ${SIZE_THRESHOLD_MB} MB found. Nothing to do."
    exit 0
  fi

  blue "📋 Paths to be removed from history (${count} unique):"
  cat /tmp/paths_to_remove.txt
  echo ""

  # ── Build filter-repo args ────────────────────────────────────────
  local args
  mapfile -t args < <(build_filter_args /tmp/paths_to_remove.txt)
  echo ""

  # ── Backup ────────────────────────────────────────────────────────
  if [[ "$BACKUP_ENABLED" == "yes" ]]; then
    local backup_dir="$HOME/git_backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    blue "📦 Creating backup at: $backup_dir"
    git bundle create "$backup_dir/repo_before_cleanup.bundle" --all >/dev/null
    cp -r .git "$backup_dir/.git_backup" 2>/dev/null || true
    green "✅ Backup done."
  else
    yellow "⚠️  Backup disabled (BACKUP_ENABLED=no)."
  fi
  echo ""

  # ── Confirmation ──────────────────────────────────────────────────
  if [[ "$AUTO_CONFIRM" != "yes" ]]; then
    read -rp "Proceed with rewriting history? (yes/no): " ans
    if [[ "$ans" != "yes" ]]; then
      yellow "❌ Cancelled."
      exit 0
    fi
  fi

  # ── Save remote URL (filter-repo removes it) ──────────────────────
  local saved_remote=""
  if [[ -z "$REMOTE_URL" ]]; then
    saved_remote="$(git remote get-url origin 2>/dev/null || true)"
  else
    saved_remote="$REMOTE_URL"
  fi

  # ── Run filter-repo ───────────────────────────────────────────────
  blue "🧹 Running git-filter-repo..."
  git filter-repo --force "${args[@]}" --invert-paths
  green "✅ History rewritten."
  echo ""

  # ── Restore remote ────────────────────────────────────────────────
  if [[ -n "$saved_remote" ]]; then
    git remote remove origin 2>/dev/null || true
    git remote add origin "$saved_remote"
    green "🔗 Remote restored: $saved_remote"
  fi
  echo ""

  # ── Write .gitignore if missing ───────────────────────────────────
  if [[ ! -f .gitignore ]]; then
    blue "📝 Creating .gitignore to prevent re-adding large files..."
    cat > .gitignore <<'EOF'
# Large data & model files
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
*.pkl
*.joblib
*.pt
*.pth
*.onnx
core/outputs/models/
core/outputs/figures/

# Python
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Environments
.venv/
venv/
env/
.conda/

# IDE / OS
.vscode/
.idea/
*.swp
*~
.DS_Store
Thumbs.db

# Logs / backups
*.log
archive/
archive_*/
*_backup/
git_backup_*/

# Secrets
.env
.env.*
*.key
*.pem
EOF
    git add .gitignore
    git commit -m "chore: add .gitignore to prevent re-adding large files" >/dev/null || true
    green "✅ .gitignore created."
  fi
  echo ""

  # ── Verify size reduction ─────────────────────────────────────────
  blue "📊 .git size after cleanup:"
  du -sh .git
  echo ""
  blue "📊 Largest remaining blobs:"
  git rev-list --objects --all \
    | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
    | awk '$1=="blob" {print $3, $4}' \
    | sort -rn | head -10 \
    | awk '{printf "%.2f MB\t%s\n", $1/1024/1024, $2}'
  echo ""

  # ── Push prompt ───────────────────────────────────────────────────
  if [[ "$AUTO_CONFIRM" != "yes" ]]; then
    read -rp "Force-push to origin/$BRANCH now? (yes/no): " push_ans
    if [[ "$push_ans" == "yes" ]]; then
      git push origin "$BRANCH" --force
      green "✅ Pushed."
    else
      yellow "⚠️  Push skipped. To push later:"
      echo "   git push origin $BRANCH --force"
    fi
  fi

  echo ""
  green "════════════════════════════════════════════════════════════════"
  green "  ✅ CLEANUP COMPLETE"
  green "════════════════════════════════════════════════════════════════"
}

main "$@"
