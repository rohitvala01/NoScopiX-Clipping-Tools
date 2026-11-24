export interface SubmitJobRequest {
  source: string;
  type: "url" | "file";
  clip_count: number;
  clip_duration: { min: number; max: number };
  captions: boolean;
  preset?: string;
  language?: string;
  range?: { start: number; end: number } | null;
  live_minutes?: number | null;
  font?: string;
  title_overlay?: boolean;
  hindi_quality?: "fast" | "best";
  theme?: "all" | "fighting" | "comedy" | "romance";
}

export interface ProbeResult {
  source: string;
  type: "url" | "file";
  title: string;
  duration: number | null;
  is_live: boolean;
  thumbnail?: string | null;
  channel?: string | null;
  has_heatmap?: boolean;
  live_status?: string | null;
  processing?: boolean; // just-ended stream YouTube hasn't processed yet
  error?: string;
}

export async function probeSource(source: string): Promise<ProbeResult> {
  const res = await fetch(`/api/probe?source=${encodeURIComponent(source)}`);
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.error) throw new Error(data.error || "Could not read this video");
  return data;
}

export async function uploadVideo(file: File, onProgress?: (pct: number) => void): Promise<string> {
  // XHR so we can show upload progress for big files
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/upload?name=${encodeURIComponent(file.name)}`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(Math.round((e.loaded / e.total) * 100));
    xhr.onload = () => {
      try {
        const data = JSON.parse(xhr.responseText);
        xhr.status === 200 && data.path ? resolve(data.path) : reject(new Error(data.error || "Upload failed"));
      } catch {
        reject(new Error("Upload failed"));
      }
    };
    xhr.onerror = () => reject(new Error("Upload failed"));
    xhr.send(file);
  });
}

export interface ClipResult {
  id: string;
  filename: string;
  path: string;
  url: string;
  score: number;
  layout?: string;
  title: string;
  hook_line: string;
  start: number;
  end: number;
  duration: number;
  viral_factors: string[];
  reason: string;
}

export interface JobStatus {
  status: "queued" | "processing" | "completed" | "failed";
  stage?: string;
  progress?: number;
  clips?: ClipResult[];
  error?: string;
  created_at?: string;
}

export async function submitJob(data: SubmitJobRequest): Promise<{ jobId: string }> {
  const res = await fetch("/api/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || "Failed to submit video for clipping");
  }
  return res.json();
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`/api/status?jobId=${encodeURIComponent(jobId)}`);
  if (!res.ok) {
    throw new Error("Failed to fetch job status");
  }
  return res.json();
}
