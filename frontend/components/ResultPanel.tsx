"use client";

import { AnalysisResponse } from "@/lib/types";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

interface Props {
  result: AnalysisResponse;
}

const COLORS = ["#F05454", "#7ECF99"];

function verdictTone(verdict: AnalysisResponse["verdict"]): string {
  if (verdict === "fake") return "var(--danger)";
  if (verdict === "suspicious") return "var(--warning)";
  return "var(--safe)";
}

export function ResultPanel({ result }: Props) {
  const pieData = [
    { name: "Fake Risk", value: Number((result.risk_score * 100).toFixed(2)) },
    { name: "Real Likelihood", value: Number(((1 - result.risk_score) * 100).toFixed(2)) },
  ];

  return (
    <section className="result-card">
      <div className="result-header">
        <h2>Analysis Verdict</h2>
        <span className="pill" style={{ borderColor: verdictTone(result.verdict), color: verdictTone(result.verdict) }}>
          {result.verdict.toUpperCase()}
        </span>
      </div>
      <p>{result.summary}</p>

      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={230}>
          <PieChart>
            <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} innerRadius={52}>
              {pieData.map((_, index) => (
                <Cell key={index} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip formatter={(value: number) => `${value}%`} />
          </PieChart>
        </ResponsiveContainer>
      </div>

      <div className="metrics-grid">
        <article>
          <h3>Risk Score</h3>
          <strong>{(result.risk_score * 100).toFixed(1)}%</strong>
        </article>
        <article>
          <h3>Confidence</h3>
          <strong>{(result.confidence * 100).toFixed(1)}%</strong>
        </article>
      </div>

      <h3>Recommendations</h3>
      <ul>
        {result.recommendations.map((rec) => (
          <li key={rec}>{rec}</li>
        ))}
      </ul>

      <h3>Modality Evidence</h3>
      <div className="modality-stack">
        {result.modality_results.map((m) => (
          <article key={`${m.modality}-${m.confidence}`} className="modality-item">
            <header>
              <span>{m.modality.toUpperCase()}</span>
              <strong>{m.verdict.toUpperCase()}</strong>
            </header>
            <p>Fake score {(m.score_fake * 100).toFixed(1)}% | confidence {(m.confidence * 100).toFixed(1)}%</p>
            <ul>
              {m.evidence.map((ev) => (
                <li key={ev}>{ev}</li>
              ))}
            </ul>
          </article>
        ))}
      </div>
    </section>
  );
}
