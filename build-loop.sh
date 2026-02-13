#!/usr/bin/env bash
set -euo pipefail

MAX_SESSIONS="${1:-0}"           # 0 = unlimited (iterate until completion)
SESSION_TIMEOUT="${2:-3600}"     # per-session timeout in seconds (default: 1 hour)
SAFETY_CAP=50                    # absolute max to prevent runaway (even in unlimited mode)
LOG_DIR=".build-sessions"
PROGRESS_FILE="claude-progress.txt"
REQUIREMENTS_FILE="docs/requirements.md"

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Count features
# ---------------------------------------------------------------------------
count_done() {
  local n
  n=$(grep -c '^\- \[x\]' "$REQUIREMENTS_FILE" 2>/dev/null || true)
  echo "${n:-0}"
}

count_total() {
  local n
  n=$(grep -c '^\- \[' "$REQUIREMENTS_FILE" 2>/dev/null || true)
  echo "${n:-0}"
}

# ---------------------------------------------------------------------------
# Session prompt (shared by all sessions)
# ---------------------------------------------------------------------------
SESSION_PROMPT=$(cat <<'PROMPT'
You are a headless build agent for the Transskribo project. Your job is to
implement one session's worth of features, test them, and commit.

## Step 0: Check for dirty state

Run `git status`. If there are uncommitted changes from a crashed prior session:
- Read the changed files to understand what was in progress.
- Run `uv run pytest` to see if the existing code works.
- If tests pass: stage the changes, commit with a message describing what was
  recovered, and mark the corresponding [ ] items as [x] in requirements.md.
- If tests fail: run `git checkout -- .` to discard the broken partial state
  and start the feature fresh.

## Step 1: Orient yourself

1. Read CLAUDE.md — understand the stack, conventions, constraints, and rules.
2. Read claude-progress.txt — see what previous sessions accomplished.
3. Read docs/tasks.md — find the NEXT session whose features are not all done.
   That is YOUR session. Note its feature list and verification criteria.
4. Read docs/requirements.md — find the [ ] items that belong to your session.
5. Read docs/design.md — understand how your features fit architecturally.

If ALL features in ALL sessions are already marked [x], reply with exactly
"ALL SESSIONS COMPLETE" on its own line and stop.

## Step 2: Install dependencies

Run `uv sync` to ensure the environment is up to date.

## Step 3: Implement features

For each [ ] feature in your session, in order:

1. Implement the feature in the correct module per the project layout.
2. Follow all conventions in CLAUDE.md:
   - Type hints on all functions
   - pathlib.Path everywhere, no raw strings
   - WhisperX/torch imports only in transcriber.py
   - Config passed as dataclass, never global
3. Write or update tests in the corresponding test file.
4. Run the checks after each feature:
   - `uv run pytest` — all tests must pass
   - `uv run ruff check src/ tests/` — no lint errors
   - `uv run pyright src/` — no type errors
5. If a check fails, fix the issue before moving on.
6. Update docs/requirements.md — change [ ] to [x] for the completed item.

## Step 4: Commit

After implementing all features for your session:

1. Stage only the files you changed (never `git add -A`).
2. Commit with a descriptive message summarizing what was built.
   Do NOT add any AI attribution to the commit message.
3. Verify with `git status` that the working tree is clean.

## Step 5: Hand off

1. Update claude-progress.txt with an entry in this exact format:

   ## YYYY-MM-DD HH:MM — Session N: <title from tasks.md>
   Features: <comma-separated requirement IDs completed>
   Status: success | partial
   Notes:
     - <what was done>
     - <what was done>
   Issues:
     - <any problems encountered, or "none">

2. Commit this update separately: "Update progress log for session N"

3. Run the session's verification criteria from docs/tasks.md.
   Report pass/fail for each criterion.

## Rules

- Implement ONLY the features listed for your session — do not work ahead.
- If you encounter a blocking error you cannot resolve after 3 attempts,
  mark the feature as partial in the progress log, commit what works, and stop.
- Never skip tests. Every feature must have a test.
- Never force-push. Never amend previous commits.
- Never hardcode tokens, paths, or secrets.
- Do not create files outside the project layout defined in CLAUDE.md.
PROMPT
)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
if [ "$MAX_SESSIONS" -eq 0 ]; then
  LIMIT_LABEL="unlimited"
else
  LIMIT_LABEL="$MAX_SESSIONS"
fi

echo "========================================"
echo " Transskribo Build Loop"
echo " Max sessions: $LIMIT_LABEL (safety cap: $SAFETY_CAP)"
echo " Session timeout: ${SESSION_TIMEOUT}s"
echo " Features: $(count_done)/$(count_total) complete"
echo "========================================"

SESSION_NUM=0
while true; do
  SESSION_NUM=$((SESSION_NUM + 1))

  # Enforce max sessions if set, and always enforce safety cap
  if [ "$MAX_SESSIONS" -gt 0 ] && [ "$SESSION_NUM" -gt "$MAX_SESSIONS" ]; then
    break
  fi
  if [ "$SESSION_NUM" -gt "$SAFETY_CAP" ]; then
    echo ""
    echo "ERROR: Reached safety cap ($SAFETY_CAP sessions). Stopping."
    exit 1
  fi

  SESSION_LOG="$LOG_DIR/session-$(date +%Y%m%d-%H%M%S).log"
  DONE_BEFORE=$(count_done)

  echo ""
  if [ "$MAX_SESSIONS" -gt 0 ]; then
    echo "=== Session $SESSION_NUM/$MAX_SESSIONS — $(date) ==="
  else
    echo "=== Session $SESSION_NUM — $(date) ==="
  fi
  echo "    Features before: $DONE_BEFORE/$(count_total)"
  echo "    Log: $SESSION_LOG"
  echo ""

  # Run the headless session, capture output (with timeout)
  set +e
  timeout "$SESSION_TIMEOUT" \
    claude --print --dangerously-skip-permissions -p "$SESSION_PROMPT" \
    2>&1 | tee "$SESSION_LOG"
  exit_code=${PIPESTATUS[0]}
  set -e

  # timeout returns 124 when the command is killed
  if [ $exit_code -eq 124 ]; then
    echo ""
    echo "ERROR: Session $SESSION_NUM timed out after ${SESSION_TIMEOUT}s. Stopping."
    exit 1
  fi

  DONE_AFTER=$(count_done)
  echo ""
  echo "=== Session $SESSION_NUM finished ==="
  echo "    Exit code: $exit_code"
  echo "    Features: $DONE_BEFORE → $DONE_AFTER / $(count_total)"
  echo "    Log: $SESSION_LOG"

  # Check for completion signal in session output
  if grep -q "ALL SESSIONS COMPLETE" "$SESSION_LOG" 2>/dev/null; then
    echo ""
    echo "========================================"
    echo " All sessions complete!"
    echo " Total features: $(count_done)/$(count_total)"
    echo "========================================"
    exit 0
  fi

  # Bail if session failed
  if [ $exit_code -ne 0 ]; then
    echo ""
    echo "ERROR: Session $SESSION_NUM exited with code $exit_code. Stopping."
    exit $exit_code
  fi

  # Bail if no progress was made (stuck session)
  if [ "$DONE_AFTER" -le "$DONE_BEFORE" ]; then
    echo ""
    echo "WARNING: No features completed in session $SESSION_NUM."
    echo "Check $SESSION_LOG for errors. Stopping to avoid infinite loop."
    exit 1
  fi

  # Check if all features are done
  if [ "$(count_done)" -eq "$(count_total)" ]; then
    echo ""
    echo "========================================"
    echo " All features complete after session $SESSION_NUM!"
    echo "========================================"
    exit 0
  fi
done

echo ""
echo "========================================"
echo " Reached max sessions ($MAX_SESSIONS)."
echo " Features: $(count_done)/$(count_total) complete"
echo "========================================"
