#!/usr/bin/env bash
# watch-progress.sh — monitor build progress from another terminal
# Usage: ./watch-progress.sh          (run once)
#        ./watch-progress.sh --loop   (refresh every 10s)

set -euo pipefail

REQUIREMENTS="docs/requirements.md"
PROGRESS="claude-progress.txt"
LOG_DIR=".build-sessions"

show_status() {
  clear

  echo "========================================"
  echo " Transskribo Build Progress"
  echo " $(date)"
  echo "========================================"

  # ── Feature counts ──────────────────────────────────────────────────────
  echo ""
  TOTAL=$(grep -c '^\- \[' "$REQUIREMENTS" 2>/dev/null || true)
  TOTAL=${TOTAL:-0}
  DONE=$(grep -c '^\- \[x\]' "$REQUIREMENTS" 2>/dev/null || true)
  DONE=${DONE:-0}
  TODO=$((TOTAL - DONE))
  if [ "$TOTAL" -gt 0 ]; then
    PCT=$((DONE * 100 / TOTAL))
  else
    PCT=0
  fi

  # Progress bar
  BAR_WIDTH=40
  FILLED=$((PCT * BAR_WIDTH / 100))
  EMPTY=$((BAR_WIDTH - FILLED))
  BAR=""
  if [ "$FILLED" -gt 0 ]; then BAR=$(printf '%0.s#' $(seq 1 $FILLED)); fi
  if [ "$EMPTY" -gt 0 ]; then BAR="${BAR}$(printf '%0.s-' $(seq 1 $EMPTY))"; fi
  echo "  Features: $DONE/$TOTAL ($PCT%)"
  echo "  [$BAR]"
  echo "  Remaining: $TODO"

  # ── Per-section breakdown ───────────────────────────────────────────────
  echo ""
  echo "── Per Section ──────────────────────────"
  echo ""
  current_section=""
  section_done=0
  section_total=0

  while IFS= read -r line; do
    if [[ "$line" =~ ^##\  ]]; then
      # Print previous section
      if [ -n "$current_section" ]; then
        printf "  %-40s %d/%d\n" "$current_section" "$section_done" "$section_total"
      fi
      current_section="${line## }"
      current_section="${current_section### }"
      section_done=0
      section_total=0
    elif [[ "$line" =~ ^\-\ \[x\] ]]; then
      section_total=$((section_total + 1))
      section_done=$((section_done + 1))
    elif [[ "$line" =~ ^\-\ \[\ \] ]]; then
      section_total=$((section_total + 1))
    fi
  done < "$REQUIREMENTS"
  # Print last section
  if [ -n "$current_section" ]; then
    printf "  %-40s %d/%d\n" "$current_section" "$section_done" "$section_total"
  fi

  # ── Recent commits ──────────────────────────────────────────────────────
  echo ""
  echo "── Recent Commits ───────────────────────"
  echo ""
  git log --oneline -10 2>/dev/null || echo "  (no commits yet)"

  # ── Last progress entry ─────────────────────────────────────────────────
  echo ""
  echo "── Last Progress Entry ──────────────────"
  echo ""
  if [ -f "$PROGRESS" ]; then
    # Extract last entry (from last ## heading to end of file)
    last_entry=$(awk '/^## /{start=NR; content=""} {if(start) content=content"\n"$0} END{print content}' "$PROGRESS")
    if [ -n "$last_entry" ]; then
      echo "$last_entry" | head -12
    else
      echo "  (no entries yet)"
    fi
  else
    echo "  (no progress file)"
  fi

  # ── Active session log ──────────────────────────────────────────────────
  if [ -d "$LOG_DIR" ]; then
    latest_log=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
      echo ""
      echo "── Latest Session Log (last 5 lines) ────"
      echo "   $latest_log"
      echo ""
      tail -5 "$latest_log" 2>/dev/null | sed 's/^/  /'
    fi
  fi

  echo ""
  echo "========================================"
}

if [ "${1:-}" = "--loop" ]; then
  while true; do
    show_status
    sleep 10
  done
else
  show_status
fi
