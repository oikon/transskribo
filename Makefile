.PHONY: spec build watch status clean

SHELL := /bin/bash
REQUIREMENTS := docs/requirements.md

# Open interactive Claude for spec editing
spec:
	claude

# Run the headless build loop (default: up to 7 sessions)
build:
	./build-loop.sh

# Run build loop for a specific number of sessions
build-%:
	./build-loop.sh $*

# Watch progress in a loop (run in another terminal)
watch:
	./watch-progress.sh --loop

# One-shot status check
status:
	@TOTAL=$$(grep -c '^\- \[' $(REQUIREMENTS) || true); \
	DONE=$$(grep -c '^\- \[x\]' $(REQUIREMENTS) || true); \
	TOTAL=$${TOTAL:-0}; DONE=$${DONE:-0}; \
	TODO=$$((TOTAL - DONE)); \
	if [ "$$TOTAL" -gt 0 ]; then PCT=$$((DONE * 100 / TOTAL)); else PCT=0; fi; \
	echo ""; \
	echo "Features: $$DONE/$$TOTAL ($$PCT%) — $$TODO remaining"; \
	echo ""
	@echo "Remaining:"
	@grep '^\- \[ \]' $(REQUIREMENTS) 2>/dev/null | sed 's/- \[ \] /  /' | head -10 || true
	@REMAINING=$$(grep -c '^\- \[ \]' $(REQUIREMENTS) || true); \
	REMAINING=$${REMAINING:-0}; \
	if [ "$$REMAINING" -gt 10 ]; then echo "  ... and $$((REMAINING - 10)) more"; fi
	@echo ""
	@echo "Recent commits:"
	@git log --oneline -5 2>/dev/null || echo "  (no commits)"
	@echo ""

# Remove build session logs
clean:
	rm -rf .build-sessions/
