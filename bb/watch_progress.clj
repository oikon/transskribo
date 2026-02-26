(ns watch-progress
  (:require [utils :as u]
            [clojure.string :as str]))

(defn show-status []
  (let [done (u/count-done)
        total (u/count-total)
        todo (- total done)
        pct (u/progress-pct)
        c u/colorize]

    ;; Header
    (println (c :bold-cyan "========================================"))
    (println (c :bold-cyan " Transskribo Build Progress"))
    (println (c :gray (str " " (u/now-timestamp))))
    (println (c :bold-cyan "========================================"))

    ;; Feature counts + progress bar
    (println)
    (let [count-color (if (= done total) :bold-green :bold-white)]
      (println (c count-color (str "  Features: " done "/" total " (" pct "%)"))))
    (println (str "  " (u/progress-bar pct)))
    (if (pos? todo)
      (println (c :yellow (str "  Remaining: " todo)))
      (println (c :green (str "  Remaining: " todo))))

    ;; Per-section breakdown
    (println)
    (println (c :cyan "── Per Section ──────────────────────────"))
    (println)
    (doseq [{:keys [section done total]} (u/per-section-breakdown)]
      (let [label (str "  " section)
            padding (- 42 (count label))
            pad (if (pos? padding) (apply str (repeat padding " ")) " ")
            color (if (= done total) :green :yellow)]
        (println (str label pad (c color (str done "/" total))))))

    ;; Recent commits
    (println)
    (println (c :cyan "── Recent Commits ───────────────────────"))
    (println)
    (println (c :gray (u/git-log-oneline 10)))

    ;; Last progress entry
    (println)
    (println (c :cyan "── Last Progress Entry ──────────────────"))
    (println)
    (if-let [entry (u/last-progress-entry)]
      (println (c :gray entry))
      (println "  (no progress file)"))

    ;; Latest session log
    (when-let [{:keys [path tail]} (u/latest-session-log)]
      (println)
      (println (c :cyan "── Latest Session Log (last 5 lines) ────"))
      (println (c :gray (str "   " path)))
      (println)
      (doseq [line tail]
        (println (c :gray (str "  " line)))))

    (println)
    (println (c :bold-cyan "========================================"))))

(defn status []
  (show-status))

(defn watch []
  (loop []
    (u/clear-screen)
    (show-status)
    (Thread/sleep 10000)
    (recur)))
