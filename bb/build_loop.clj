(ns build-loop
  (:require [babashka.fs :as fs]
            [babashka.process :as p]
            [cheshire.core :as json]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [utils :as u]))

;; ---------------------------------------------------------------------------
;; Session prompt (verbatim from build-loop.sh)
;; ---------------------------------------------------------------------------

(def session-prompt
  "You are a headless build agent for the Transskribo project. Your job is to
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
\"ALL SESSIONS COMPLETE\" on its own line and stop.

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
     - <any problems encountered, or \"none\">

2. Commit this update separately: \"Update progress log for session N\"

3. Run the session's verification criteria from docs/tasks.md.
   Report pass/fail for each criterion.

## Rules

- Implement ONLY the features listed for your session — do not work ahead.
- If you encounter a blocking error you cannot resolve after 3 attempts,
  mark the feature as partial in the progress log, commit what works, and stop.
- Never skip tests. Every feature must have a test.
- Never force-push. Never amend previous commits.
- Never hardcode tokens, paths, or secrets.
- Do not create files outside the project layout defined in CLAUDE.md.")

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn parse-args
  "Parse CLI args. Supports positional max-sessions (int) and --verbose / -v flag.
   Returns {:max-sessions N, :verbose? bool}."
  [args]
  (loop [args args
         max-sessions 50
         verbose? false]
    (if (empty? args)
      {:max-sessions max-sessions :verbose? verbose?}
      (let [arg (first args)]
        (cond
          (or (= arg "--verbose") (= arg "-v"))
          (recur (rest args) max-sessions true)

          :else
          (let [n (try (Integer/parseInt arg)
                       (catch Exception _ nil))]
            (recur (rest args)
                   (if (and n (pos? n)) n max-sessions)
                   verbose?)))))))

(defn check-completion [log-path]
  (when (fs/exists? log-path)
    (str/includes? (slurp (str log-path)) "ALL SESSIONS COMPLETE")))

;; ---------------------------------------------------------------------------
;; Stream-JSON event display
;; ---------------------------------------------------------------------------

;; Track whether the last output was inline text (no trailing newline),
;; so we can insert a newline before tool_use / result lines.
(def ^:private needs-newline? (atom false))

(defn- ensure-newline!
  "Print a newline if the previous output was inline text without one."
  []
  (when @needs-newline?
    (println)
    (reset! needs-newline? false)))

(defn- display-event
  "Parse a stream-json line and print a human-readable summary to the console.
   Returns the raw line for logging."
  [line]
  (let [c u/colorize]
    (try
      (let [event (json/parse-string line true)]
        (case (:type event)
          "assistant"
          (doseq [block (get-in event [:message :content] [])]
            (case (:type block)
              "tool_use"  (do (ensure-newline!)
                              (println (c :cyan (str "    > " (:name block)
                                                     (when-let [cmd (get-in block [:input :command])]
                                                       (str ": " (first (str/split-lines cmd))))))))
              "text"      (let [text (:text block)]
                            (print text)
                            (flush)
                            (reset! needs-newline? (not (str/ends-with? text "\n"))))
              nil))

          "result"
          (let [turns  (:num_turns event)
                cost   (:total_cost_usd event)
                millis (:duration_ms event)]
            (ensure-newline!)
            (println (c :gray (str "    ✓ Done — "
                                   turns " turns, "
                                   (when millis (str (quot millis 1000) "s, "))
                                   (when cost (format "$%.2f" (double cost)))))))

          ;; system, hook events — skip silently
          nil))
      (catch Exception _
        ;; Not JSON — print raw (shouldn't happen with stream-json)
        (ensure-newline!)
        (println line))))
  line)

;; ---------------------------------------------------------------------------
;; Run a single session
;; ---------------------------------------------------------------------------

(defn run-session [session-num max-sessions verbose?]
  (let [c u/colorize
        log-path (str u/log-dir "/session-" (u/now-log-suffix) ".log")
        done-before (u/count-done)
        total (u/count-total)]

    (println)
    (println (c :bold-cyan (str "=== Session " session-num "/" max-sessions " — " (u/now-timestamp) " ===")))
    (println (c :bold-white (str "    Features before: " done-before "/" total)))
    (println (c :gray (str "    Log: " log-path)))
    (println)

    ;; Reset newline tracker for each session
    (reset! needs-newline? false)

    ;; Verbose: stream-json with parsed event display
    ;; Quiet:   plain text passthrough
    (let [cmd (if verbose?
                ["claude" "--print" "--verbose" "--output-format" "stream-json"
                 "--dangerously-skip-permissions"]
                ["claude" "--print"
                 "--dangerously-skip-permissions"])
          proc (apply p/process {:out :stream :err :out :in ""}
                      (conj (vec cmd) session-prompt))
          line-handler (if verbose? display-event identity)
          exit-code (with-open [writer (io/writer log-path)
                                reader (io/reader (:out proc))]
                      (doseq [line (line-seq reader)]
                        (line-handler line)
                        (when-not verbose?
                          (println line))
                        (.write writer line)
                        (.write writer "\n")
                        (.flush writer))
                      (:exit @proc))
          done-after (u/count-done)]

      (println)
      (println (c :bold-cyan (str "=== Session " session-num " finished ===")))
      (let [exit-color (if (zero? exit-code) :green :bold-red)]
        (println (c exit-color (str "    Exit code: " exit-code))))
      (println (c :bold-white (str "    Features: " done-before " → " done-after " / " total)))
      (println (c :gray (str "    Log: " log-path)))

      {:exit exit-code
       :log-path log-path
       :done-before done-before
       :done-after done-after})))

;; ---------------------------------------------------------------------------
;; Main build loop
;; ---------------------------------------------------------------------------

(defn run-build-loop [args]
  (let [c u/colorize
        {:keys [max-sessions verbose?]} (parse-args args)]
    (fs/create-dirs u/log-dir)

    (println (c :bold-cyan "========================================"))
    (println (c :bold-cyan " Transskribo Build Loop"))
    (println (c :bold-white (str " Max sessions: " max-sessions
                                  (when verbose? ", verbose"))))
    (println (c :bold-white (str " Features: " (u/count-done) "/" (u/count-total) " complete")))
    (println (c :bold-cyan "========================================"))

    (loop [i 1]
      (if (> i max-sessions)
        (do
          (println)
          (println (c :bold-yellow "========================================"))
          (println (c :bold-yellow (str " Reached max sessions (" max-sessions ").")))
          (println (c :bold-white (str " Features: " (u/count-done) "/" (u/count-total) " complete")))
          (println (c :bold-yellow "========================================")))

        (let [{:keys [exit log-path done-before done-after]} (run-session i max-sessions verbose?)]
          (cond
            ;; Completion signal
            (check-completion log-path)
            (do
              (println)
              (println (c :bold-green "========================================"))
              (println (c :bold-green " All sessions complete!"))
              (println (c :bold-green (str " Total features: " (u/count-done) "/" (u/count-total))))
              (println (c :bold-green "========================================"))
              (System/exit 0))

            ;; Non-zero exit
            (not (zero? exit))
            (do
              (println)
              (println (c :bold-red (str "ERROR: Session " i " exited with code " exit ". Stopping.")))
              (System/exit exit))

            ;; No progress
            (<= done-after done-before)
            (do
              (println)
              (println (c :bold-yellow (str "WARNING: No features completed in session " i ".")))
              (println (c :bold-yellow (str "Check " log-path " for errors. Stopping to avoid infinite loop.")))
              (System/exit 1))

            ;; All features done
            (= (u/count-done) (u/count-total))
            (do
              (println)
              (println (c :bold-green "========================================"))
              (println (c :bold-green (str " All features complete after session " i "!")))
              (println (c :bold-green "========================================"))
              (System/exit 0))

            ;; Continue to next session
            :else
            (recur (inc i))))))))
