#!/bin/bash
# Wrapper script to run Python scripts from project
# Usage:
#   ./scripts/run.sh preprocess_resize.py --data-dir data/raw/train
#   ./scripts/run.sh train_ensemble.py --mode det --model 1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PADDLE_DIR="$ROOT_DIR/approaches/paddle"

if [ $# -eq 0 ]; then
    echo "Usage: ./scripts/run.sh <script_name> [args...]"
    echo ""
    echo "Scripts in scripts/:"
    ls -1 "$SCRIPT_DIR"/*.py 2>/dev/null | xargs -n1 basename || echo "  (none)"
    echo ""
    echo "Scripts in approaches/paddle/scripts/:"
    ls -1 "$PADDLE_DIR/scripts/"*.py 2>/dev/null | xargs -n1 basename || echo "  (none)"
    exit 0
fi

SCRIPT_NAME="$1"
shift

# Always use paddle environment (has all dependencies)
cd "$PADDLE_DIR"

# Check scripts/ first, then approaches/paddle/scripts/
if [ -f "$SCRIPT_DIR/$SCRIPT_NAME" ]; then
    exec uv run python "$SCRIPT_DIR/$SCRIPT_NAME" "$@"
elif [ -f "$PADDLE_DIR/scripts/$SCRIPT_NAME" ]; then
    exec uv run python "scripts/$SCRIPT_NAME" "$@"
else
    echo "Error: Script not found: $SCRIPT_NAME"
    echo "Searched in: $SCRIPT_DIR and $PADDLE_DIR/scripts/"
    exit 1
fi
