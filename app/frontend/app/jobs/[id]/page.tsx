"use client";

import React, { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeft,
  Download,
  Play,
  Pause,
  Clock,
  Sparkles,
  AlertCircle,
  CheckCircle2,
  Share2,
  FileVideo,
  Loader2
} from "lucide-react";
import { getJobStatus, ClipResult } from "@/lib/api";
import { formatDuration, getScoreColor } from "@/lib/utils";

export default function JobStatusPage() {
  const params = useParams();
  const router = useRouter();
  const jobId = params.id as string;
  const [playingClipId, setPlayingClipId] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const handleCopyLink = () => {
    if (typeof window !== "undefined") {
      navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const { data: job, isLoading, error } = useQuery({
    queryKey: ["jobStatus", jobId],
    queryFn: () => getJobStatus(jobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "completed" || status === "failed") {
        return false;
      }
      return 2500; // poll every 2.5s while processing
    },
  });

  if (isLoading && !job) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] space-y-4">
        <Loader2 className="w-8 h-8 text-indigo-500 animate-spin" />
        <p className="text-sm text-zinc-400">Loading job status...</p>
      </div>
    );
  }

  if (error || job?.status === "failed") {
    return (
      <div className="max-w-md mx-auto glass-card p-6 text-center space-y-4">
        <AlertCircle className="w-10 h-10 text-rose-500 mx-auto" />
        <h2 className="text-lg font-bold text-zinc-200">Processing Failed</h2>
        <p className="text-xs text-zinc-400">{job?.error || (error as any)?.message || "An unexpected error occurred."}</p>
        <button
          onClick={() => router.push("/")}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-xs font-semibold text-zinc-200 transition"
        >
          <ArrowLeft className="w-3.5 h-3.5" /> Back to Home
        </button>
      </div>
    );
  }

  const isCompleted = job?.status === "completed";
  const clips = job?.clips || [];

  return (
    <div className="space-y-8">
      {/* Top Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-zinc-800/80 pb-6">
        <div>
          <button
            onClick={() => router.push("/")}
            className="inline-flex items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-200 mb-2 transition"
          >
            <ArrowLeft className="w-3.5 h-3.5" /> New Clipping Job
          </button>
          <h1 className="text-2xl font-bold flex items-center gap-2.5">
            Job <span className="font-mono text-indigo-400 text-lg">#{jobId}</span>
            {isCompleted ? (
              <span className="text-[11px] font-semibold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-0.5 rounded-full flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" /> Ready
              </span>
            ) : (
              <span className="text-[11px] font-semibold text-amber-400 bg-amber-500/10 border border-amber-500/20 px-2.5 py-0.5 rounded-full flex items-center gap-1">
                <Loader2 className="w-3 h-3 animate-spin" /> Processing
              </span>
            )}
          </h1>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleCopyLink}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-zinc-800 hover:border-zinc-700 bg-zinc-900/60 hover:bg-zinc-800 text-xs font-medium text-zinc-300 transition"
          >
            <Share2 className="w-3.5 h-3.5" />
            {copied ? "Copied!" : "Share Link"}
          </button>
          {isCompleted && clips.length > 0 && (
            <div className="text-xs text-zinc-400 font-medium hidden sm:block">
              Generated <span className="text-zinc-100 font-bold">{clips.length}</span> viral clips
            </div>
          )}
        </div>
      </div>

      {/* Progress Card when running */}
      {!isCompleted && (
        <div className="glass-card p-6 sm:p-8 space-y-6 max-w-2xl mx-auto">
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-semibold text-zinc-300 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-indigo-400 animate-pulse" />
                {job?.stage || "Processing video pipeline..."}
              </span>
              <span className="font-mono font-bold text-indigo-400">{job?.progress || 10}%</span>
            </div>
            {/* Progress bar */}
            <div className="w-full bg-zinc-950 h-2.5 rounded-full overflow-hidden border border-zinc-800">
              <div
                className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-500 ease-out"
                style={{ width: `${job?.progress || 10}%` }}
              />
            </div>
          </div>

          {/* Stepper info */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2">
            {[
              { label: "Download", step: 10 },
              { label: "Transcribe", step: 30 },
              { label: "AI Virality", step: 70 },
              { label: "9:16 + Captions", step: 85 },
            ].map((s) => (
              <div
                key={s.label}
                className={`p-2.5 rounded-xl border text-center text-xs transition ${
                  (job?.progress || 0) >= s.step
                    ? "border-indigo-500/30 bg-indigo-500/10 text-indigo-300 font-semibold"
                    : "border-zinc-800/40 text-zinc-600"
                }`}
              >
                {s.label}
              </div>
            ))}
          </div>

          <p className="text-xs text-zinc-500 text-center">
            Extracting audio, computing replay heatmaps, tracking faces, and generating animated subtitles. You can leave this tab open.
          </p>
        </div>
      )}

      {/* Clips Grid */}
      {isCompleted && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {clips.map((clip, idx) => (
              <div key={clip.id} className="glass-card overflow-hidden flex flex-col justify-between group">
                {/* 9:16 Aspect Video Container */}
                <div className="relative aspect-[9/16] bg-zinc-950 flex items-center justify-center border-b border-zinc-800/80">
                  <video
                    src={clip.url || clip.path}
                    controls
                    className="w-full h-full object-cover"
                    playsInline
                    preload="metadata"
                  />
                  {/* Virality Score Badge */}
                  <div
                    className={`absolute top-3 left-3 px-2.5 py-1 rounded-full text-xs font-bold border backdrop-blur-md shadow-md ${getScoreColor(
                      clip.score
                    )}`}
                  >
                    ★ {clip.score} Viral Score
                  </div>

                  {/* Duration & Layout Badges */}
                  <div className="absolute top-3 right-3 flex items-center gap-1.5">
                    <span className="px-2 py-0.5 rounded-md bg-black/75 backdrop-blur-sm text-[10px] font-medium text-zinc-300 border border-white/10">
                      {clip.layout === "split_cam"
                        ? "🎮 Split-Cam"
                        : clip.layout === "single_speaker"
                        ? "🎙️ Solo Speaker"
                        : "✨ 9:16 Fit"}
                    </span>
                    <span className="px-2 py-0.5 rounded-md bg-black/75 backdrop-blur-sm text-[11px] font-mono text-zinc-300 border border-white/10">
                      {formatDuration(clip.duration)}
                    </span>
                  </div>
                </div>

                {/* Clip Info Card */}
                <div className="p-4 space-y-3 flex-1 flex flex-col justify-between">
                  <div className="space-y-1.5">
                    <h3 className="font-bold text-sm text-zinc-100 line-clamp-1 group-hover:text-indigo-400 transition">
                      {clip.title}
                    </h3>
                    <p className="text-xs italic text-zinc-400 line-clamp-2">
                      &ldquo;{clip.hook_line}&rdquo;
                    </p>
                  </div>

                  {/* Tags */}
                  {clip.viral_factors && clip.viral_factors.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 pt-1">
                      {clip.viral_factors.map((f) => (
                        <span
                          key={f}
                          className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-zinc-800/80 text-zinc-400 border border-zinc-700/50"
                        >
                          #{f}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Action Buttons */}
                  <div className="pt-2 border-t border-zinc-800/60 flex items-center gap-2">
                    <a
                      href={clip.url || clip.path}
                      download={`${clip.filename || `clip_${clip.id}.mp4`}`}
                      className="flex-1 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs flex items-center justify-center gap-1.5 shadow-sm transition"
                    >
                      <Download className="w-3.5 h-3.5" /> Download MP4
                    </a>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
