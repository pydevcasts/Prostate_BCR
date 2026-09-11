#!/bin/bash
# =====================================================================
# clean_big_files.sh (v2 — robust, cross-platform)
# Dynamically find and remove large files from git history.
#
# Improvements over v1:
#   - Fixed awk parsing bug (MB vs path)
#   - Removed mapfile dependency (bash 3+ compatible)
#   - Auto-fallback to filter-branch if filter-repo crashes
#   - Better error messages
#   - Works on Python 3.10 – 3.14
# =====================================================================

set -uo pipefail   # NOTE: removed -e so we can catch filter-repo failure

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
SIZE_THRESHOLD_MB="${SIZE_THRESHOLD_MB:-5}"
BACKUP_ENABLED="${BACKUP_ENABLED:-yes}"
AUTO_CONFIRM="${AUTO_CONFIRM:-no}"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
REMOTE_URL="${REMOTE_URL:-}"
FORCE_METHOD="${FORCE_METHOD:-auto}"   # auto | filter-repo | filter-branch

# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
red()    { printf "\033[31m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
blue()   { printf "\033[34m%s\033[0m\n" "$*"; }

require_git() {
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    red "❌ Not inside a git repository."; exit 1; }
}

require_clean_tree() {
  if [[ -n "$(git status --porcelain)" ]]; then
    yellow "⚠️  Working tree is not clean."
    echo "   Commit, stash, or set AUTO_CONFIRM=yes to force."
    [[ "$AUTO_CONFIRM" != "yes" ]] && exit 1
  fi
}

# ─────────────────────────────────────────────────────────────────────
# DETECTION
# ─────────────────────────────────────────────────────────────────────
find_large_files_in_head() {
  local t=$(( SIZE_THRESHOLD_MB * 1024 * 1024 ))
  git ls-tree -r -l HEAD \
    | awk -v t="$t" '$4+0 >= t {print $4, $5}' \
    | sort -rn \
    | awk '{printf "%.2f MB\t%s\n", $1/1024/1024, $2}'
}

find_large_blobs_anywhere() {
  local t=$(( SIZE_THRESHOLD_MB * 1024 * 1024 ))
  git rev-list --objects --all \
    | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
    | awk -v t="$t" '$1=="blob" && $3+0 >= t {print $3, $4}' \
    | sort -rn \
    | awk '!seen[$2]++ {printf "%.2f MB\t%s\n", $1/1024/1024, $2}'
}

# Extract path (2nd tab-separated field) from a line "123.45 MB\tpath"
extract_path() {
  printf '%s\n' "$1" | awk -F'\t' '{print $2}'
}

# ─────────────────────────────────────────────────────────────────────
# METHOD 1 — filter-repo (fast, but may crash on Python 3.14)
# ─────────────────────────────────────────────────────────────────────
try_filter_repo() {
  local list_file="$1"
  blue "🧹 Trying git-filter-repo..."

  # Build args safely (avoid mapfile for bash 3 compat)
  local args=""
  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    args="$args --path \"$path\""
  done < "$list_file"

  # Eval to expand the args string into proper argv
  if eval "git filter-repo --force $args --invert-paths"; then
    green "✅ filter-repo succeeded."
    return 0
  else
    yellow "⚠️  filter-repo failed (likely Python version incompatibility)."
    return 1
  fi
}

# ─────────────────────────────────────────────────────────────────────
# METHOD 2 — filter-branch (slow, but works everywhere)
# ─────────────────────────────────────────────────────────────────────
try_filter_branch() {
  local list_file="$1"
  blue "🧹 Falling back to git filter-branch (slower but compatible)..."

  # Build the rm command for the index-filter
  local rm_cmd="git rm --cached --ignore-unmatch"
  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    rm_cmd="$rm_cmd '$path'"
  done < "$list_file"

  git filter-branch --force --index-filter "$rm_cmd" \
    --prune-empty --tag-name-filter cat -- --all

  # Post-cleanup
  rm -rf .git/refs/original/ 2>/dev/null || true
  git reflog expire --expire=now --all 2>/dev/null || true
  git gc --prune=now 2>/dev/null || true

  green "✅ filter-branch finished."
  return 0
}

# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
main() {
  require_git
  require_clean_tree

  yellow "════════════════════════════════════════════════════════════════"
  yellow "  GIT HISTORY CLEANUP — Dynamic Large-File Remover (v2)"
  yellow "════════════════════════════════════════════════════════════════"
  echo "Repository : $(pwd)"
  echo "Branch     : $BRANCH"
  echo "Threshold  : ${SIZE_THRESHOLD_MB} MB"
  echo "Method     : $FORCE_METHOD"
  echo ""

  # ── Scan ──────────────────────────────────────────────────────────
  blue "🔍 Large files in current HEAD (>= ${SIZE_THRESHOLD_MB} MB):"
  find_large_files_in_head | tee /tmp/large_head.txt || true
  echo ""

  blue "🔍 Large blobs anywhere in history (>= ${SIZE_THRESHOLD_MB} MB):"
  find_large_blobs_anywhere | tee /tmp/large_history.txt || true
  echo ""

  # ── Extract paths (FIXED: awk -F'\t' $2) ─────────────────────────
  : > /tmp/paths_to_remove.txt
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    p="$(extract_path "$line")"
    [[ -n "$p" ]] && echo "$p" >> /tmp/paths_to_remove.txt
  done < /tmp/large_history.txt

  sort -u /tmp/paths_to_remove.txt -o /tmp/paths_to_remove.txt

  local count
  count=$(wc -l < /tmp/paths_to_remove.txt | tr -d ' ')
  if [[ "$count" -eq 0 ]]; then
    green "✅ No files >= ${SIZE_THRESHOLD_MB} MB found. Nothing to do."
    exit 0
  fi

  blue "📋 Paths to be removed from history (${count} unique):"
  cat /tmp/paths_to_remove.txt
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

  # ── Confirm ───────────────────────────────────────────────────────
  if [[ "$AUTO_CONFIRM" != "yes" ]]; then
    read -rp "Proceed with rewriting history? (yes/no): " ans
    [[ "$ans" != "yes" ]] && { yellow "❌ Cancelled."; exit 0; }
  fi

  # ── Save remote ───────────────────────────────────────────────────
  local saved_remote
  if [[ -z "$REMOTE_URL" ]]; then
    saved_remote="$(git remote get-url origin 2>/dev/null || true)"
  else
    saved_remote="$REMOTE_URL"
  fi

  # ── Execute (with fallback) ───────────────────────────────────────
  local method_ok=0
  case "$FORCE_METHOD" in
    filter-repo)
      try_filter_repo /tmp/paths_to_remove.txt && method_ok=1
      ;;
    filter-branch)
      try_filter_branch /tmp/paths_to_remove.txt && method_ok=1
      ;;
    auto|*)
      if try_filter_repo /tmp/paths_to_remove.txt; then
        method_ok=1
      else
        try_filter_branch /tmp/paths_to_remove.txt && method_ok=1
      fi
      ;;
  esac

  if [[ "$method_ok" -ne 1 ]]; then
    red "❌ Both methods failed. Nothing was changed."
    red "   Consider the 'nuclear' approach: fresh repo, copy files, init, push."
    exit 1
  fi

  # ── Restore remote ────────────────────────────────────────────────
  if [[ -n "$saved_remote" ]]; then
    git remote remove origin 2>/dev/null || true
    git remote add origin "$saved_remote"
    green "🔗 Remote restored: $saved_remote"
  fi
  echo ""

  # ── Write .gitignore if missing ───────────────────────────────────
  if [[ ! -f .gitignore ]]; then
    blue "📝 Creating .gitignore..."
    cat > .gitignore <<'EOF'
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
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/
env/
.conda/
.vscode/
.idea/
*.swp
*~
.DS_Store
Thumbs.db
*.log
archive/
archive_*/
*_backup/
git_backup_*/
.env
.env.*
*.key
*.pem
EOF
    git add .gitignore
    git commit -m "chore: add .gitignore to prevent re-adding large files" >/dev/null 2>&1 || true
    green "✅ .gitignore created."
  fi
  echo ""

  # ── Verify ────────────────────────────────────────────────────────
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

  # ── Push ──────────────────────────────────────────────────────────
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