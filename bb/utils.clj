(ns utils
  (:require [babashka.fs :as fs]
            [babashka.process :as p]
            [clojure.string :as str])
  (:import [java.time LocalDateTime]
           [java.time.format DateTimeFormatter]))

;; ---------------------------------------------------------------------------
;; Constants
;; ---------------------------------------------------------------------------

(def requirements-file "docs/requirements.md")
(def progress-file "claude-progress.txt")
(def log-dir ".build-sessions")

;; ---------------------------------------------------------------------------
;; Requirements parsing
;; ---------------------------------------------------------------------------

(defn- read-requirements []
  (when (fs/exists? requirements-file)
    (slurp requirements-file)))

(defn count-done []
  (let [content (read-requirements)]
    (if content
      (count (re-seq #"(?m)^- \[x\]" content))
      0)))

(defn count-total []
  (let [content (read-requirements)]
    (if content
      (count (re-seq #"(?m)^- \[" content))
      0)))

(defn remaining-items []
  (let [content (read-requirements)]
    (if content
      (->> (str/split-lines content)
           (filter #(re-matches #"^- \[ \] .*" %))
           (mapv #(str/replace % #"^- \[ \] " "")))
      [])))

(defn per-section-breakdown []
  (let [content (read-requirements)]
    (if content
      (let [lines (str/split-lines content)]
        (loop [lines lines
               current-section nil
               section-done 0
               section-total 0
               result []]
          (if (empty? lines)
            (if current-section
              (conj result {:section current-section :done section-done :total section-total})
              result)
            (let [line (first lines)
                  rest-lines (rest lines)]
              (cond
                (re-matches #"^## .*" line)
                (let [section-name (str/replace line #"^## " "")
                      result (if current-section
                               (conj result {:section current-section :done section-done :total section-total})
                               result)]
                  (recur rest-lines section-name 0 0 result))

                (re-matches #"^- \[x\].*" line)
                (recur rest-lines current-section (inc section-done) (inc section-total) result)

                (re-matches #"^- \[ \].*" line)
                (recur rest-lines current-section section-done (inc section-total) result)

                :else
                (recur rest-lines current-section section-done section-total result))))))
      [])))

;; ---------------------------------------------------------------------------
;; ANSI colors
;; ---------------------------------------------------------------------------

(def ansi
  {:reset       "\033[0m"
   :bold        "\033[1m"
   :dim         "\033[2m"
   :red         "\033[31m"
   :green       "\033[32m"
   :yellow      "\033[33m"
   :cyan        "\033[36m"
   :bold-red    "\033[1;31m"
   :bold-green  "\033[1;32m"
   :bold-yellow "\033[1;33m"
   :bold-cyan   "\033[1;36m"
   :bold-white  "\033[1;37m"
   :gray        "\033[90m"})

(defn colorize [& color-text-pairs]
  (let [parts (partition 2 color-text-pairs)]
    (str (apply str (map (fn [[color text]]
                           (str (get ansi color "") text))
                         parts))
         (:reset ansi))))

;; ---------------------------------------------------------------------------
;; Progress display helpers
;; ---------------------------------------------------------------------------

(defn progress-pct []
  (let [total (count-total)
        done (count-done)]
    (if (zero? total) 0 (int (* 100 (/ done total))))))

(defn progress-bar
  ([] (progress-bar (progress-pct) 40))
  ([pct] (progress-bar pct 40))
  ([pct width]
   (let [filled (int (* pct width 0.01))
         empty  (- width filled)]
     (str "[" (:green ansi) (apply str (repeat filled "#"))
          (:gray ansi) (apply str (repeat empty "-"))
          (:reset ansi) "]"))))

;; ---------------------------------------------------------------------------
;; Terminal helpers
;; ---------------------------------------------------------------------------

(defn clear-screen []
  (print "\033[2J\033[H")
  (flush))

;; ---------------------------------------------------------------------------
;; Timestamps
;; ---------------------------------------------------------------------------

(defn now-timestamp []
  (.format (LocalDateTime/now) (DateTimeFormatter/ofPattern "yyyy-MM-dd HH:mm:ss")))

(defn now-log-suffix []
  (.format (LocalDateTime/now) (DateTimeFormatter/ofPattern "yyyyMMdd-HHmmss")))

;; ---------------------------------------------------------------------------
;; Git helpers
;; ---------------------------------------------------------------------------

(defn git-log-oneline [n]
  (try
    (-> (p/process {:out :string :err :string} "git" "log" "--oneline" (str "-" n))
        deref
        :out
        str/trim)
    (catch Exception _
      "(no commits yet)")))

;; ---------------------------------------------------------------------------
;; Progress file helpers
;; ---------------------------------------------------------------------------

(defn last-progress-entry []
  (when (fs/exists? progress-file)
    (let [content (slurp progress-file)
          lines (str/split-lines content)
          ;; Find all ## heading indices
          heading-indices (keep-indexed (fn [i line]
                                          (when (str/starts-with? line "## ")
                                            i))
                                        lines)]
      (when (seq heading-indices)
        (let [last-idx (last heading-indices)
              entry-lines (subvec (vec lines) last-idx)]
          (->> (take 12 entry-lines)
               (str/join "\n")))))))

;; ---------------------------------------------------------------------------
;; Session log helpers
;; ---------------------------------------------------------------------------

(defn latest-session-log []
  (when (fs/exists? log-dir)
    (let [logs (->> (fs/glob log-dir "*.log")
                    (sort-by #(fs/last-modified-time %) #(compare %2 %1)))]
      (when (seq logs)
        (let [path (str (first logs))
              content (slurp path)
              lines (str/split-lines content)
              tail (take-last 5 lines)]
          {:path path
           :tail (vec tail)})))))
