import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import fs from "fs";
import path from "path";
import { sendJobToBackend, startBackend, getPythonCmd } from "@/lib/backend";

declare global {
  var _jobStore: Map<string, any> | undefined;
}

const jobStore = global._jobStore || new Map<string, any>();
if (process.env.NODE_ENV !== "production") {
  global._jobStore = jobStore;
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { source, type, clip_count, clip_duration, captions, preset, language,
            range, live_minutes, font, title_overlay, hindi_quality, theme } = body;
    const sceneTheme = ["all", "fighting", "comedy", "romance"].includes(theme) ? theme : "all";
    const hindiQuality = hindi_quality === "best" ? "best" : "fast";
    const num = (v: any) => (v === null || v === undefined || v === "" || isNaN(Number(v)) ? null : Number(v));
    const clipCount = Math.max(2, Math.floor(num(clip_count) ?? 6));
    const minDur = Math.max(1, num(clip_duration?.min) ?? 15);
    const maxDur = Math.min(180, Math.max(minDur + 1, num(clip_duration?.max) ?? 30));
    const rangeStart = num(range?.start);
    const rangeEnd = num(range?.end);
    const liveMinutes = num(live_minutes);
    const fontName = typeof font === "string" && /^[\w ]{1,40}$/.test(font) ? font : "";
    const lang = /^[a-z]{2,3}$|^auto$/.test(String(language || "")) ? String(language) : "auto";

    if (!source) {
      return NextResponse.json({ error: "Missing source video URL or path" }, { status: 400 });
    }

    const jobId = Math.random().toString(36).substring(2, 10);
    const jobData = {
      job_id: jobId,
      source,
      source_type: type || "url",
      clip_count: clipCount,
      clip_min_duration: minDur,
      clip_max_duration: maxDur,
      range_start: rangeStart,
      range_end: rangeEnd,
      live_minutes: liveMinutes,
      caption_font: fontName || null,
      title_overlay: title_overlay !== false,
      hindi_quality: hindiQuality,
      captions: captions !== false,
      caption_preset: preset || "streamer",
      language: lang,
      theme: sceneTheme,
      status: "processing",
      stage: "Downloading source video & engagement heatmaps...",
      progress: 10,
      clips: [],
      created_at: new Date().toISOString(),
    };

    jobStore.set(jobId, jobData);

    const backendUrl = process.env.MODAL_API_URL;
    if (backendUrl) {
      // Modal serverless cloud dispatch
      fetch(`${backendUrl}/submit_job`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(jobData),
      }).catch((err) => {
        console.error("Failed to forward job to Modal backend:", err);
      });
    } else if (await sendJobToBackend(jobData)) {
      // Fast path: the always-on backend already has the speech model loaded
      console.log(`[Job ${jobId}] Sent to always-on backend`);
    } else {
      // Backend not running yet: start it for next time, and run this job as a one-off process
      startBackend();
      const backendDir = path.resolve(process.cwd(), "..", "backend");
      const appScript = path.join(backendDir, "app.py");

      const args = [
        appScript,
        source,
        "--job-id", jobId,
        "--clips", String(clipCount),
        "--min", String(minDur),
        "--max", String(maxDur),
        "--language", lang,
        "--theme", sceneTheme,
      ];
      if (rangeStart !== null && rangeEnd !== null && rangeEnd > rangeStart) {
        args.push("--start", String(rangeStart), "--end", String(rangeEnd));
      }
      if (liveMinutes !== null && liveMinutes > 0) args.push("--live-minutes", String(liveMinutes));
      if (fontName) args.push("--font", fontName);
      if (title_overlay === false) args.push("--no-title");
      args.push("--hindi-quality", hindiQuality);

      if (captions === false) {
        args.push("--no-captions");
      } else if (preset) {
        args.push("--preset", preset);
      }

      const jobLogDir = path.join(backendDir, "data", jobId);
      if (!fs.existsSync(jobLogDir)) {
        fs.mkdirSync(jobLogDir, { recursive: true });
      }
      const logFd = fs.openSync(path.join(jobLogDir, "process.log"), "a");

      console.log(`[Job ${jobId}] Spawning local Python process: ${getPythonCmd()} ${args.join(" ")}`);
      const child = spawn(getPythonCmd(), args, {
        cwd: backendDir,
        detached: true,
        stdio: ["ignore", logFd, logFd],
      });
      child.unref();
    }

    return NextResponse.json({ jobId });
  } catch (error: any) {
    console.error("API submit error:", error);
    return NextResponse.json({ error: error.message || "Internal server error" }, { status: 500 });
  }
}
