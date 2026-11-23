"use client";

import React, { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Link2, Upload, Sparkles, Type, Loader2, Search, Clock, Radio, Scissors, Minus, Plus, Film, Flame,
} from "lucide-react";
import { submitJob, probeSource, uploadVideo, ProbeResult } from "@/lib/api";

// ── time helpers ────────────────────────────────────────────────────────────────
function fmtTime(sec: number): string {
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${m}:${String(s).padStart(2, "0")}`;
}

function parseTime(text: string): number | null {
  const parts = text.trim().split(":").map((p) => p.trim());
  if (!parts.length || parts.some((p) => p === "" || isNaN(Number(p)))) return null;
  return parts.reduce((acc, p) => acc * 60 + Number(p), 0);
}

function TimeInput({ value, onCommit, label }: { value: number; onCommit: (v: number) => void; label: string }) {
  const [text, setText] = useState(fmtTime(value));
  useEffect(() => setText(fmtTime(value)), [value]);
  const commit = () => {
    const v = parseTime(text);
    if (v === null) setText(fmtTime(value));
    else onCommit(v);
  };
  return (
    <label className="flex flex-col gap-1 text-[10px] uppercase tracking-wider text-zinc-500 font-semibold">
      {label}
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), commit())}
        className="w-28 bg-zinc-950 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-sm text-zinc-100 font-mono text-center focus:outline-none focus:border-indigo-500"
      />
    </label>
  );
}

// ── options ─────────────────────────────────────────────────────────────────────
const LENGTH_PRESETS = [
  { id: "quick", label: "Quick", sub: "5–15s", min: 5, max: 15 },
  { id: "shorts", label: "Shorts", sub: "15–30s", min: 15, max: 30 },
  { id: "reels", label: "Reels", sub: "30–60s", min: 30, max: 60 },
  { id: "long", label: "Long", sub: "60–90s", min: 60, max: 90 },
  { id: "custom", label: "Custom", sub: "your range", min: 0, max: 0 },
];

const STYLES = [
  { id: "streamer", label: "🎮 Streamer", desc: "Yellow 1-3 words, pop" },
  { id: "hormozi", label: "🔥 Hormozi", desc: "Yellow active word + reveal" },
  { id: "beast", label: "💥 Beast", desc: "One huge word, colour cycle" },
  { id: "neon", label: "⚡ Neon", desc: "Glowing lime outline" },
  { id: "boxed", label: "⬛ Boxed", desc: "White on black box" },
  { id: "karaoke", label: "🎤 Karaoke", desc: "Words fill as spoken" },
  { id: "fire", label: "🔥 Fire", desc: "Gold + red outline" },
  { id: "meme", label: "😂 Meme", desc: "Comic font, 3D shadow" },
  { id: "clean", label: "✨ Clean", desc: "Minimal soft shadow" },
];

const FONTS = [
  { id: "", label: "Style default", css: "inherit" },
  { id: "Arial Black", label: "Arial Black", css: "'Arial Black', sans-serif" },
  { id: "Impact", label: "Impact", css: "Impact, sans-serif" },
  { id: "Segoe UI Black", label: "Segoe UI Black", css: "'Segoe UI Black', sans-serif" },
  { id: "Comic Sans MS", label: "Comic Sans", css: "'Comic Sans MS', cursive" },
  { id: "Bahnschrift", label: "Bahnschrift", css: "Bahnschrift, sans-serif" },
  { id: "Franklin Gothic Medium", label: "Franklin Gothic", css: "'Franklin Gothic Medium', sans-serif" },
  { id: "Verdana", label: "Verdana", css: "Verdana, sans-serif" },
];

export default function HomePage() {
  const router = useRouter();
  const fileRef = useRef<HTMLInputElement>(null);

  // source + probe
  const [sourceInput, setSourceInput] = useState("");
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [probing, setProbing] = useState(false);
  const [uploadPct, setUploadPct] = useState<number | null>(null);

  // range / live
  const [rangeStart, setRangeStart] = useState(0);
  const [rangeEnd, setRangeEnd] = useState(0);
  const [liveMinutes, setLiveMinutes] = useState(10);

  // clip settings
  const [clipCount, setClipCount] = useState(6);
  const [lengthPreset, setLengthPreset] = useState("shorts");
  const [minDuration, setMinDuration] = useState(15);
  const [maxDuration, setMaxDuration] = useState(30);
  const [captions, setCaptions] = useState(true);
  const [titleOverlay, setTitleOverlay] = useState(true);
  const [captionPreset, setCaptionPreset] = useState("streamer");
  const [font, setFont] = useState("");
  const [language, setLanguage] = useState("auto");
  const [hindiQuality, setHindiQuality] = useState<"fast" | "best">("fast");

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const duration = probe?.duration ?? 0;
  const isLive = !!probe?.is_live;
  const selectedLen = isLive ? liveMinutes * 60 : Math.max(0, rangeEnd - rangeStart);
  const isFullRange = !isLive && rangeStart <= 0.5 && rangeEnd >= duration - 0.5;
  const suggestedMax = Math.max(2, Math.floor(selectedLen / (minDuration + 8)));

  const analyze = async (src?: string) => {
    const raw = typeof src === "string" ? src : sourceInput;
    const value = (raw ?? "").trim().replace(/^"|"$/g, "");
    if (!value) return setError("Paste a video link or a local file path first.");
    setError(null);
    setProbing(true);
    setProbe(null);
    try {
      const info = await probeSource(value);
      setProbe(info);
      setSourceInput(info.source || value);
      setRangeStart(0);
      setRangeEnd(info.duration ?? 0);
    } catch (e: any) {
      setError(e.message || "Could not read this video");
    } finally {
      setProbing(false);
    }
  };

  const onFilePicked = async (file: File) => {
    setError(null);
    setProbe(null);
    setUploadPct(0);
    try {
      const path = await uploadVideo(file, setUploadPct);
      setSourceInput(path);
      await analyze(path);
    } catch (e: any) {
      setError(e.message || "Upload failed");
    } finally {
      setUploadPct(null);
    }
  };

  const pickLength = (id: string) => {
    setLengthPreset(id);
    const p = LENGTH_PRESETS.find((x) => x.id === id);
    if (p && p.id !== "custom") {
      setMinDuration(p.min);
      setMaxDuration(p.max);
    }
  };

  const setStart = (v: number) => setRangeStart(Math.max(0, Math.min(v, rangeEnd - 5)));
  const setEnd = (v: number) => setRangeEnd(Math.min(duration, Math.max(v, rangeStart + 5)));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!probe) return setError("Analyze the video first so we know its length.");
    if (minDuration < 1 || maxDuration > 180 || minDuration >= maxDuration) {
      return setError("Clip length must be between 1 and 180 seconds, with min below max.");
    }
    if (clipCount < 2) return setError("Pick at least 2 clips.");
    if (selectedLen < minDuration) return setError("The selected part of the video is shorter than one clip.");

    setIsLoading(true);
    try {
      const res = await submitJob({
        source: probe.source,
        type: probe.type,
        clip_count: clipCount,
        clip_duration: { min: minDuration, max: maxDuration },
        captions,
        preset: captionPreset,
        language,
        range: !isLive && !isFullRange ? { start: Math.round(rangeStart), end: Math.round(rangeEnd) } : null,
        live_minutes: isLive ? liveMinutes : null,
        font,
        title_overlay: titleOverlay,
        hindi_quality: hindiQuality,
      });
      router.push(`/jobs/${res.jobId}`);
    } catch (err: any) {
      setError(err.message || "Failed to submit job");
      setIsLoading(false);
    }
  };

  const pctStart = duration ? (rangeStart / duration) * 100 : 0;
  const pctEnd = duration ? (rangeEnd / duration) * 100 : 100;

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      {/* Hero */}
      <div className="text-center space-y-4">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-indigo-500/20 bg-indigo-500/10 text-indigo-400 text-xs font-semibold uppercase tracking-wider">
          <Sparkles className="w-3.5 h-3.5" /> Multimodal Engagement AI
        </div>
        <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight">
          Turn Long Videos into <span className="gradient-text">Viral Shorts</span>
        </h1>
        <p className="text-zinc-400 max-w-xl mx-auto text-sm sm:text-base">
          Videos, finished streams and live streams. Pick the part you want, and get 9:16 shorts with Hinglish captions and titles.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="glass-card p-6 sm:p-8 space-y-7">
        {/* ── 1. Source ── */}
        <section className="space-y-3">
          <StepLabel n={1} icon={<Link2 className="w-4 h-4 text-indigo-400" />} text="Video" />
          <div className="flex gap-2">
            <input
              type="text"
              placeholder="YouTube / Twitch link, live stream link, or a local file path"
              value={sourceInput}
              onChange={(e) => {
                setSourceInput(e.target.value);
                setProbe(null);
              }}
              onPaste={(e) => {
                const text = e.clipboardData.getData("text");
                if (text) {
                  setSourceInput(text);
                  setTimeout(() => analyze(text), 0);
                }
              }}
              onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), analyze())}
              className="flex-1 min-w-0 bg-zinc-950/80 border border-zinc-800 rounded-xl px-4 py-3 text-sm placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500"
            />
            <button
              type="button"
              onClick={() => analyze()}
              disabled={probing}
              className="px-4 rounded-xl bg-indigo-500/90 hover:bg-indigo-500 text-white text-sm font-semibold flex items-center gap-1.5 disabled:opacity-60"
            >
              {probing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
              Analyze
            </button>
          </div>
          <div className="flex items-center gap-3 text-xs text-zinc-500">
            <span>or</span>
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              disabled={uploadPct !== null}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-zinc-800 hover:border-zinc-700 text-zinc-300"
            >
              <Upload className="w-3.5 h-3.5" /> Upload a video file
            </button>
            {uploadPct !== null && <span className="text-indigo-400">Uploading… {uploadPct}%</span>}
            <input
              ref={fileRef}
              type="file"
              accept="video/mp4,video/quicktime,video/webm,video/x-matroska,.mkv"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && onFilePicked(e.target.files[0])}
            />
          </div>

          {probing && (
            <div className="text-xs text-zinc-400 flex items-center gap-2">
              <Loader2 className="w-3.5 h-3.5 animate-spin" /> Reading video length…
            </div>
          )}

          {probe && (
            <div className="flex gap-3 items-center p-3 rounded-xl border border-zinc-800 bg-zinc-950/50">
              {probe.thumbnail ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={probe.thumbnail} alt="" className="w-28 h-16 object-cover rounded-lg flex-none" />
              ) : (
                <div className="w-28 h-16 rounded-lg bg-zinc-900 flex items-center justify-center flex-none">
                  <Film className="w-6 h-6 text-zinc-600" />
                </div>
              )}
              <div className="min-w-0 space-y-1">
                <div className="text-sm font-semibold text-zinc-100 truncate">{probe.title}</div>
                <div className="flex flex-wrap gap-1.5 text-[11px]">
                  {isLive ? (
                    <span className="px-2 py-0.5 rounded-full bg-rose-500/15 text-rose-400 font-bold flex items-center gap-1">
                      <Radio className="w-3 h-3" /> LIVE NOW
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-300 flex items-center gap-1">
                      <Clock className="w-3 h-3" /> {fmtTime(duration)}
                    </span>
                  )}
                  {probe.channel && <span className="px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-400">{probe.channel}</span>}
                  {probe.has_heatmap && (
                    <span className="px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-400 flex items-center gap-1">
                      <Flame className="w-3 h-3" /> Most Replayed data
                    </span>
                  )}
                </div>
              </div>
            </div>
          )}
        </section>

        {probe?.processing && (
          <div className="p-3 text-xs text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-xl space-y-1">
            <div className="font-bold">⏳ YouTube is still processing this stream</div>
            <div className="text-amber-200/80">
              This live stream ended recently. Until YouTube finishes processing it (for very long streams
              this can take hours or more), it downloads extremely slowly or not at all. Please try again later,
              or use a saved copy of the video.
            </div>
          </div>
        )}

        {/* ── 2. Range / live recording ── */}
        {probe && (
          <section className="space-y-3 pt-5 border-t border-zinc-800/80">
            <StepLabel
              n={2}
              icon={isLive ? <Radio className="w-4 h-4 text-rose-400" /> : <Scissors className="w-4 h-4 text-purple-400" />}
              text={isLive ? "How much of the live stream to record" : "Part of the video to clip"}
            />
            {isLive ? (
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-zinc-400">Record the next</span>
                  <span className="text-rose-400 font-bold">{liveMinutes} min</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={120}
                  value={liveMinutes}
                  onChange={(e) => setLiveMinutes(Number(e.target.value))}
                  className="w-full accent-rose-500"
                  aria-label="Minutes of live stream to record"
                />
                <p className="text-[11px] text-zinc-500">
                  Recording starts when you press Generate; clips are made once the recording finishes.
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="dual-range relative h-6">
                  <div className="absolute top-1/2 -translate-y-1/2 left-0 right-0 h-1.5 rounded-full bg-zinc-800" />
                  <div
                    className="absolute top-1/2 -translate-y-1/2 h-1.5 rounded-full bg-gradient-to-r from-indigo-500 to-purple-500"
                    style={{ left: `${pctStart}%`, right: `${100 - pctEnd}%` }}
                  />
                  <input
                    type="range"
                    min={0}
                    max={Math.max(1, Math.round(duration))}
                    value={Math.round(rangeStart)}
                    onChange={(e) => setStart(Number(e.target.value))}
                    aria-label="Start time"
                  />
                  <input
                    type="range"
                    min={0}
                    max={Math.max(1, Math.round(duration))}
                    value={Math.round(rangeEnd)}
                    onChange={(e) => setEnd(Number(e.target.value))}
                    aria-label="End time"
                  />
                </div>
                <div className="flex flex-wrap items-end justify-between gap-3">
                  <div className="flex items-end gap-3">
                    <TimeInput label="From" value={rangeStart} onCommit={setStart} />
                    <span className="text-zinc-600 pb-2">→</span>
                    <TimeInput label="To" value={rangeEnd} onCommit={setEnd} />
                  </div>
                  <div className="text-right">
                    <div className="text-[10px] uppercase tracking-wider text-zinc-500 font-semibold">Selected</div>
                    <div className="text-sm font-bold text-indigo-300">{fmtTime(selectedLen)}</div>
                  </div>
                  {selectedLen > 3 * 3600 && (
                    <p className="basis-full text-[11px] text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2">
                      That's {fmtTime(selectedLen)} of video. It will work but take a while
                      (roughly {Math.round((selectedLen / 3600) * 1.5)}–{Math.round((selectedLen / 3600) * 2.5)} min).
                      For quick results, pick the best 1–2 hours with the slider.
                    </p>
                  )}
                  {!isFullRange && (
                    <button
                      type="button"
                      onClick={() => (setRangeStart(0), setRangeEnd(duration))}
                      className="text-xs px-3 py-1.5 rounded-lg border border-zinc-800 hover:border-zinc-700 text-zinc-300"
                    >
                      Use full video
                    </button>
                  )}
                </div>
              </div>
            )}
          </section>
        )}

        {/* ── 3. Clips ── */}
        <section className="space-y-4 pt-5 border-t border-zinc-800/80">
          <StepLabel n={3} icon={<Film className="w-4 h-4 text-emerald-400" />} text="Clips" />
          <div className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-6 items-start">
            <div className="space-y-2">
              <div className="text-xs text-zinc-400 font-medium">Number of clips</div>
              <div className="flex items-center gap-2">
                <button type="button" aria-label="Fewer clips" onClick={() => setClipCount((c) => Math.max(2, c - 1))}
                  className="w-9 h-9 rounded-lg border border-zinc-800 hover:border-zinc-700 flex items-center justify-center">
                  <Minus className="w-4 h-4" />
                </button>
                <input
                  type="number"
                  min={2}
                  value={clipCount}
                  onChange={(e) => setClipCount(Math.max(2, Math.floor(Number(e.target.value) || 2)))}
                  className="w-16 h-9 bg-zinc-950 border border-zinc-800 rounded-lg text-center text-sm font-bold focus:outline-none focus:border-indigo-500"
                  aria-label="Number of clips"
                />
                <button type="button" aria-label="More clips" onClick={() => setClipCount((c) => c + 1)}
                  className="w-9 h-9 rounded-lg border border-zinc-800 hover:border-zinc-700 flex items-center justify-center">
                  <Plus className="w-4 h-4" />
                </button>
              </div>
              <div className="text-[10px] text-zinc-500">
                Min 2.{probe && selectedLen > 0 && ` This part fits ~${suggestedMax}.`}
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-xs text-zinc-400 font-medium">
                Clip length: <span className="text-indigo-300 font-bold">{minDuration}–{maxDuration}s</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {LENGTH_PRESETS.map((p) => (
                  <button
                    type="button"
                    key={p.id}
                    onClick={() => pickLength(p.id)}
                    className={`px-3 py-1.5 rounded-lg border text-left transition-all ${
                      lengthPreset === p.id
                        ? "border-indigo-500 bg-indigo-500/15 ring-1 ring-indigo-500/50"
                        : "border-zinc-800 bg-zinc-950/40 hover:border-zinc-700"
                    }`}
                  >
                    <div className="text-xs font-bold text-zinc-200">{p.label}</div>
                    <div className="text-[10px] text-zinc-500">{p.sub}</div>
                  </button>
                ))}
              </div>
              {lengthPreset === "custom" && (
                <div className="flex items-center gap-2 pt-1">
                  <input type="number" min={1} max={179} value={minDuration} aria-label="Minimum clip seconds"
                    onChange={(e) => setMinDuration(Math.max(1, Math.min(179, Number(e.target.value))))}
                    className="w-20 bg-zinc-950 border border-zinc-800 rounded-lg px-2 py-1.5 text-xs text-center focus:outline-none focus:border-indigo-500" />
                  <span className="text-zinc-600 text-xs">to</span>
                  <input type="number" min={2} max={180} value={maxDuration} aria-label="Maximum clip seconds"
                    onChange={(e) => setMaxDuration(Math.max(2, Math.min(180, Number(e.target.value))))}
                    className="w-20 bg-zinc-950 border border-zinc-800 rounded-lg px-2 py-1.5 text-xs text-center focus:outline-none focus:border-indigo-500" />
                  <span className="text-zinc-500 text-[11px]">seconds (max 180)</span>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* ── 4. Look ── */}
        <section className="space-y-4 pt-5 border-t border-zinc-800/80">
          <StepLabel n={4} icon={<Type className="w-4 h-4 text-yellow-400" />} text="Captions & title" />
          <div className="flex flex-wrap gap-5">
            <Toggle checked={captions} onChange={setCaptions} label="Animated captions" />
            <Toggle checked={titleOverlay} onChange={setTitleOverlay} label="Title on video" />
          </div>

          {captions && (
            <>
              <div className="space-y-2">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400">Caption style</div>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
                  {STYLES.map((p) => (
                    <button
                      type="button"
                      key={p.id}
                      onClick={() => setCaptionPreset(p.id)}
                      className={`p-3 rounded-xl border text-left transition-all ${
                        captionPreset === p.id
                          ? "border-indigo-500 bg-indigo-500/15 ring-1 ring-indigo-500/50"
                          : "border-zinc-800/80 bg-zinc-950/40 hover:border-zinc-700"
                      }`}
                    >
                      <div className="text-xs font-bold text-zinc-200">{p.label}</div>
                      <div className="text-[10px] text-zinc-400 mt-0.5">{p.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400">Caption font</div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                  {FONTS.map((f) => (
                    <button
                      type="button"
                      key={f.label}
                      onClick={() => setFont(f.id)}
                      className={`p-2.5 rounded-xl border text-left transition-all ${
                        font === f.id
                          ? "border-indigo-500 bg-indigo-500/15 ring-1 ring-indigo-500/50"
                          : "border-zinc-800/80 bg-zinc-950/40 hover:border-zinc-700"
                      }`}
                    >
                      <div className="text-sm text-yellow-300 truncate" style={{ fontFamily: f.css, fontWeight: 800 }}>
                        BHAI KYA SCENE
                      </div>
                      <div className="text-[10px] text-zinc-400 mt-0.5">{f.label}</div>
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}

          <div className="space-y-2">
            <label htmlFor="language" className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400">
              Video language
            </label>
            <select
              id="language"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="w-full p-2.5 rounded-xl border border-zinc-800/80 bg-zinc-950/40 text-sm text-zinc-200"
            >
              <option value="auto">Auto-detect (Hinglish or English)</option>
              <option value="hi">Hindi / Hinglish (captions in English letters)</option>
              <option value="en">English</option>
              <option value="mr">Marathi (मराठी)</option>
              <option value="bn">Bengali (বাংলা)</option>
              <option value="pa">Punjabi (ਪੰਜਾਬੀ)</option>
              <option value="gu">Gujarati (ગુજરાતી)</option>
              <option value="ta">Tamil (தமிழ்)</option>
              <option value="te">Telugu (తెలుగు)</option>
              <option value="kn">Kannada (ಕನ್ನಡ)</option>
              <option value="ml">Malayalam (മലയാളം)</option>
              <option value="ur">Urdu (اردو)</option>
            </select>
          </div>

          {language !== "en" && (
            <div className="space-y-2">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400">
                Hindi / Hinglish caption quality
              </div>
              <div className="grid grid-cols-2 gap-2.5">
                {[
                  { id: "fast" as const, label: "⚡ Fast", desc: "Quick results, good captions" },
                  { id: "best" as const, label: "🎯 Best", desc: "Most accurate Hindi, ~4x slower transcription" },
                ].map((q) => (
                  <button
                    type="button"
                    key={q.id}
                    onClick={() => setHindiQuality(q.id)}
                    className={`p-3 rounded-xl border text-left transition-all ${
                      hindiQuality === q.id
                        ? "border-indigo-500 bg-indigo-500/15 ring-1 ring-indigo-500/50"
                        : "border-zinc-800/80 bg-zinc-950/40 hover:border-zinc-700"
                    }`}
                  >
                    <div className="text-xs font-bold text-zinc-200">{q.label}</div>
                    <div className="text-[10px] text-zinc-400 mt-0.5">{q.desc}</div>
                  </button>
                ))}
              </div>
              <p className="text-[10px] text-zinc-500">English videos always use the fast mode, which is already accurate.</p>
            </div>
          )}
        </section>

        {error && (
          <div className="p-3 text-xs text-rose-400 bg-rose-500/10 border border-rose-500/20 rounded-xl">{error}</div>
        )}

        <button
          type="submit"
          disabled={isLoading || !probe || !!probe?.processing}
          className="w-full py-4 rounded-xl font-bold text-sm tracking-wide bg-gradient-to-r from-indigo-500 via-purple-500 to-indigo-600 hover:opacity-95 text-white shadow-lg shadow-indigo-500/25 flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" /> Starting…
            </>
          ) : !probe ? (
            <>Analyze a video to continue</>
          ) : probe.processing ? (
            <>Stream still processing on YouTube, try later</>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              {isLive
                ? `Record ${liveMinutes} min & make ${clipCount} clips`
                : `Make ${clipCount} clips from ${fmtTime(selectedLen)}`}
            </>
          )}
        </button>
      </form>
    </div>
  );
}

function StepLabel({ n, icon, text }: { n: number; icon: React.ReactNode; text: string }) {
  return (
    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-zinc-300">
      <span className="w-5 h-5 rounded-full bg-zinc-800 text-zinc-400 text-[10px] flex items-center justify-center">{n}</span>
      {icon}
      {text}
    </div>
  );
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <label className="flex items-center gap-2 text-xs text-zinc-300 cursor-pointer select-none">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="w-4 h-4 rounded accent-indigo-500"
      />
      {label}
    </label>
  );
}
