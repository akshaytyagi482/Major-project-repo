import { AnalysisResponse } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

export async function analyzeText(text: string): Promise<AnalysisResponse> {
  const response = await fetch(`${API_BASE}/api/v1/analyze/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export async function analyzeUrl(url: string): Promise<AnalysisResponse> {
  const response = await fetch(`${API_BASE}/api/v1/analyze/url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export async function analyzeFile(file: File): Promise<AnalysisResponse> {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch(`${API_BASE}/api/v1/analyze/file`, {
    method: "POST",
    body,
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export async function analyzeMultimodal(text: string, files: File[]): Promise<AnalysisResponse> {
  const body = new FormData();
  if (text.trim()) {
    body.append("text", text.trim());
  }
  files.forEach((file) => body.append("files", file));

  const response = await fetch(`${API_BASE}/api/v1/analyze/multimodal`, {
    method: "POST",
    body,
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export async function downloadReport(result: AnalysisResponse): Promise<Blob> {
  const response = await fetch(`${API_BASE}/api/v1/report/pdf`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(result),
  });

  if (!response.ok) {
    throw new Error("Failed to generate PDF report.");
  }
  return response.blob();
}
