"use client";

import { FormEvent, useMemo, useState } from "react";
import { motion } from "framer-motion";

import { ResultPanel } from "@/components/ResultPanel";
import { analyzeFile, analyzeMultimodal, analyzeText, analyzeUrl, downloadReport } from "@/lib/api";
import { AnalysisResponse } from "@/lib/types";

type Mode = "text" | "url" | "file" | "multimodal";

export default function HomePage() {
  const [mode, setMode] = useState<Mode>("multimodal");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);

  const title = useMemo(() => {
    if (mode === "text") return "Analyze News Text";
    if (mode === "url") return "Analyze News URL";
    if (mode === "file") return "Analyze Single File";
    return "Analyze Text + Media Together";
  }, [mode]);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);

    try {
      let output: AnalysisResponse;
      if (mode === "text") {
        output = await analyzeText(text);
      } else if (mode === "url") {
        output = await analyzeUrl(url);
      } else if (mode === "file") {
        if (!files[0]) throw new Error("Select one file for file mode.");
        output = await analyzeFile(files[0]);
      } else {
        output = await analyzeMultimodal(text, files);
      }
      setResult(output);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  }

  async function handleReportDownload() {
    if (!result) return;
    const blob = await downloadReport(result);
    const href = window.URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = href;
    anchor.download = "detection_report.pdf";
    anchor.click();
    window.URL.revokeObjectURL(href);
  }

  return (
    <main className="page-shell">
      <motion.section
        className="hero"
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7 }}
      >
        <h1>TruthLens</h1>
        <p>
          Multimodal fake news and deepfake detection platform for journalists, educators, legal teams, and fact-checkers.
        </p>
      </motion.section>

      <motion.section
        className="panel"
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1, duration: 0.6 }}
      >
        <div className="mode-switch">
          {(["multimodal", "text", "url", "file"] as Mode[]).map((m) => (
            <button key={m} type="button" onClick={() => setMode(m)} className={mode === m ? "active" : ""}>
              {m.toUpperCase()}
            </button>
          ))}
        </div>

        <h2>{title}</h2>

        <form onSubmit={handleSubmit} className="form-stack">
          {(mode === "text" || mode === "multimodal") && (
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste news text, transcript, caption, or narrative here..."
              rows={7}
            />
          )}

          {mode === "url" && (
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com/news-story"
              required
            />
          )}

          {(mode === "file" || mode === "multimodal") && (
            <input
              type="file"
              multiple={mode === "multimodal"}
              onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
              accept=".jpg,.jpeg,.png,.webp,.mp4,.mov,.avi,.mkv,.webm,.txt"
            />
          )}

          <button disabled={loading} type="submit" className="submit-btn">
            {loading ? "Analyzing..." : "Run Detection"}
          </button>
        </form>

        {error && <p className="error">{error}</p>}
      </motion.section>

      {result && (
        <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
          <div className="result-actions">
            <h2>Detection Results</h2>
            <button onClick={handleReportDownload} type="button">
              Download PDF Report
            </button>
          </div>
          <ResultPanel result={result} />
        </motion.section>
      )}
    </main>
  );
}
