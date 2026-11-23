import { NextRequest, NextResponse } from "next/server";
import { execFile } from "child_process";
import path from "path";
import { BACKEND_URL, backendUp, startBackend, getPythonCmd } from "@/lib/backend";

export const runtime = "nodejs";

// Reads a video's title, length and live status before a job starts (runs backend/probe.py).
export async function GET(req: NextRequest) {
  const source = (new URL(req.url).searchParams.get("source") || "").trim();
  if (!source) {
    return NextResponse.json({ error: "Missing source" }, { status: 400 });
  }
  // Fast path: the always-on backend answers without starting a new Python process
  if (await backendUp()) {
    try {
      const res = await fetch(`${BACKEND_URL}/probe?source=${encodeURIComponent(source)}`, {
        signal: AbortSignal.timeout(120_000),
        cache: "no-store",
      });
      const data = await res.json();
      return NextResponse.json(data, { status: data.error ? 422 : 200 });
    } catch {
      // fall through to the one-off script
    }
  } else {
    startBackend(); // warm it up while the user picks options
  }

  const backendDir = path.resolve(process.cwd(), "..", "backend");
  const script = path.join(backendDir, "probe.py");

  return new Promise<NextResponse>((resolve) => {
    execFile(
      getPythonCmd(),
      [script, source],
      { cwd: backendDir, timeout: 120_000, maxBuffer: 4 * 1024 * 1024, env: { ...process.env, PYTHONIOENCODING: "utf-8" } },
      (err, stdout) => {
        try {
          const data = JSON.parse(stdout.trim().split("\n").pop() || "{}");
          resolve(NextResponse.json(data, { status: data.error ? 422 : 200 }));
        } catch {
          resolve(NextResponse.json({ error: err?.message || "Could not read this video" }, { status: 500 }));
        }
      }
    );
  });
}
