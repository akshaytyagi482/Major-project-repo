export type Verdict = "real" | "suspicious" | "fake";

export interface ModalityResult {
  modality: "text" | "image" | "video";
  score_fake: number;
  score_real: number;
  confidence: number;
  verdict: Verdict;
  evidence: string[];
  metadata: Record<string, unknown>;
}

export interface AnalysisResponse {
  verdict: Verdict;
  confidence: number;
  risk_score: number;
  summary: string;
  modality_results: ModalityResult[];
  recommendations: string[];
}
