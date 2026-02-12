---
description: Review project state, interview about changes, and route to implementation or spec update
allowed-tools: Read, Grep, Glob, Edit, Write, Bash(git add:*), Bash(git status:*), Bash(git commit:*), Bash(git log:*), Bash(uv run pytest:*), Bash(uv run ruff:*)
---

## Current project context

Design spec:
!`cat docs/design.md`

Requirements:
!`cat docs/requirements.md`

Tasks:
!`cat docs/tasks.md`

Progress log:
!`cat claude-progress.txt 2>/dev/null || echo "No progress log yet."`

Recent commits:
!`git log --oneline -30`

Test health:
!`uv run pytest --tb=short 2>&1 | tail -20`

## Instructions

Summarize the current state:
- What's complete (count and last completed feature from requirements.md)
- What's pending (count and next planned feature)
- Any failed or stuck items in the progress log
- Test suite health (from the output above)

Then ask what I want to work on next. Interview me about new features,
bugs, or refactoring — one question at a time. Continue until the scope
is fully understood.

When done, assess the scope:

**Simple change** (small bug fix, tweak to existing behavior, one-file
addition, nothing that alters architecture or adds requirement items):
- Tell me this is straightforward and doesn't need a spec update
- Ask if I want it implemented immediately

**Complex change** (new feature spanning multiple modules, architectural
shift, new CLI command, pipeline changes, or anything that would add items
to design.md or requirements.md):
- Update the relevant spec files (design.md, requirements.md, tasks.md)
- Provide a concise summary of what changed in the specs
- Do NOT implement the code — I will decide when to do that
- Stage only the spec files and commit as "spec: <description>"
