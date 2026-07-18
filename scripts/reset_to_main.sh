#!/bin/bash
# Change directory to the project root
cd "$(dirname "$0")/.." || exit 1

echo "========================================================"
echo "[WARNING] This will reset your local repository to match"
echo "the remote 'main' branch."
echo "All uncommitted changes and untracked files will be"
echo "PERMANENTLY DELETED!"
echo "========================================================"
echo ""
read -p "Are you sure you want to proceed? (Y/N): " confirm

if [[ "$confirm" != "Y" && "$confirm" != "y" ]]; then
    echo ""
    echo "Operation cancelled. Exiting safely."
    exit 0
fi

echo ""
echo "Resetting to the latest origin/main..."
# Force delete untracked large directories to prevent issues
# This is done BEFORE git reset --hard so any accidentally deleted tracked files are restored.
rm -rf preprocessed_data dataset 2>/dev/null
rm -rf venv src/__pycache__ app_back/__pycache__ 2>/dev/null


git fetch origin --prune
git reset --hard origin/main

git clean -fdx
echo ""
echo "Reset complete!"
