"""Viral Clipper Backend — Modal Serverless App & Local FastAPI Runner."""
import asyncio
import logging
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("viral_clipper")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from pipeline.download import download_video
from pipeline.transcribe import transcribe_video
from pipeline.analyze import analyze_signals, detect_scene_changes
from pipeline.reframe import get_video_layout
from pipeline.score import score_and_rank
from pipeline.export import export_all_clips

# ── Modal App Definition ──
try:
    import modal
    app = modal.App("viral-clipper")

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
        .pip_install(
            "yt-dlp>=2024.0",
            "faster-whisper>=1.0.0",
            "google-generativeai>=0.8.0",
            "mediapipe>=0.10.0",
            "ffmpeg-python>=0.2.0",
            "librosa>=0.10.0",
            "soundfile>=0.12.0",
            "scenedetect[opencv]>=0.6.0",
            "numpy>=1.24.0",
            "httpx>=0.27.0",
            "pydantic>=2.0.0",
            "opencv-python-headless>=4.8.0",
        )
    )
    secrets = [modal.Secret.from_name("viral-clipper-secrets")]
except ImportError:
    modal = None
    app = None
    image = None
    secrets = None

class JobRequest(BaseModel):
    job_id: str
    source: str
    source_type: str = "url"
    clip_count: int = 10
    clip_min_duration: float = 5.0
    clip_max_duration: float = 25.0
    captions: bool = True
    caption_preset: str = "streamer"
    language: str = "auto"
    range_start: Optional[float] = None
    range_end: Optional[float] = None
    live_minutes: Optional[float] = None
    caption_font: Optional[str] = None
    title_overlay: bool = True
    hindi_quality: str = "fast"
    webhook_url: Optional[str] = None

# In-memory status store for local execution
_job_statuses: Dict[str, Dict[str, Any]] = {}

def update_status(job_id: str, stage: str, progress: int, status: str = "processing", clips: Optional[List] = None, error: Optional[str] = None):
    data = {
        "status": status,
        "stage": stage,
        "progress": progress,
        "clips": clips or [],
        "error": error
    }
    _job_statuses[job_id] = data
    try:
        import json
        status_file = config.DATA_DIR / job_id / "status.json"
        status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass
    logger.info(f"[{job_id}] ({progress}%) {stage}")

async def run_pipeline(
    job_id: str,
    source: str,
    source_type: str = "url",
    clip_count: int = 10,
    clip_min_duration: float = 5.0,
    clip_max_duration: float = 25.0,
    captions: bool = True,
    caption_preset: str = "streamer",
    webhook_url: Optional[str] = None,
    language: str = "auto",
    range_start: Optional[float] = None,
    range_end: Optional[float] = None,
    live_minutes: Optional[float] = None,
    caption_font: Optional[str] = None,
    title_overlay: bool = True,
    hindi_quality: str = "fast",
    theme: str = "all",
) -> Dict[str, Any]:
    """Execute end-to-end viral clipping pipeline."""
    job_dir = config.DATA_DIR / job_id
    output_dir = config.OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_count = max(config.MIN_CLIPS, int(clip_count))
    clip_min_duration = max(config.ABSOLUTE_MIN_CLIP_DURATION, float(clip_min_duration))
    clip_max_duration = min(config.ABSOLUTE_MAX_CLIP_DURATION, max(clip_min_duration + 1, float(clip_max_duration)))

    try:
        # Stage 1: Download
        if live_minutes:
            update_status(job_id, f"Recording {live_minutes:g} min of the live stream", 5)
        else:
            update_status(job_id, "Downloading source and metadata", 10)
        t0 = time.perf_counter()
        # Links (not live): audio-first. Only the audio is downloaded up front; each clip's video
        # section is fetched later. Falls back to a full download if anything goes wrong.
        audio_first = config.AUDIO_FIRST and source.strip().startswith(("http://", "https://")) and not live_minutes
        metadata = None
        if source.strip().startswith(("http://", "https://")) and not live_minutes:
            # Fail fast on streams YouTube is still processing instead of crawling for hours
            from pipeline.download import _yt_info, PROCESSING_MSG
            try:
                _info = await asyncio.to_thread(_yt_info, source)
            except Exception:
                _info = {}
            if _info.get("live_status") == "post_live":
                raise RuntimeError(PROCESSING_MSG)
        if audio_first:
            try:
                from pipeline.fast_url import fetch_audio
                update_status(job_id, "Downloading audio (video is fetched per clip)", 10)
                metadata = await fetch_audio(source, job_dir, start=range_start, end=range_end)
            except Exception as e:
                logger.warning(f"Audio-first download not possible ({e}); downloading the full video.")
                audio_first = False
        if metadata is None:
            metadata = await download_video(source, job_dir, start=range_start, end=range_end,
                                            live_minutes=live_minutes)
        if not metadata.duration or metadata.duration < 1:
            raise RuntimeError(
                "The download came back empty, so there is nothing to clip. If this is a live stream "
                "that just ended, YouTube may still be processing it; try again later or use a saved copy."
            )
        t_dl = time.perf_counter()

        # Stage 2: Transcribe (GPU) while scene + layout detection run in parallel (CPU/ffmpeg)
        update_status(job_id, "Transcribing audio with word timestamps", 30)
        if audio_first:
            scene_task = layout_task = None  # no full video to scan; webcam is found per clip
        else:
            scene_task = asyncio.create_task(asyncio.to_thread(detect_scene_changes, metadata.source_path))
            layout_task = asyncio.create_task(asyncio.to_thread(get_video_layout, metadata.source_path))
        transcript = await transcribe_video(metadata.source_path, job_dir, language=language,
                                            quality=hindi_quality)
        t_tr = time.perf_counter()

        # Stage 3: Multimodal Signal Analysis
        update_status(job_id, "Analyzing viral energy & engagement signals", 50)
        scene_changes = await scene_task if scene_task else []
        analysis = await analyze_signals(
            metadata.source_path, transcript, metadata, job_dir,
            scene_changes=scene_changes,
            min_duration=clip_min_duration,
            max_duration=clip_max_duration,
            target_count=max(config.CANDIDATE_COUNT, clip_count * 3),
            theme=theme,
        )
        t_an = time.perf_counter()

        # Stage 4: LLM Judging & Boundary Snapping
        update_status(job_id, "AI virality scoring and hook extraction", 70)
        ranked = await score_and_rank(
            candidates=analysis.candidates,
            transcript=transcript,
            clip_count=clip_count,
            min_duration=clip_min_duration,
            max_duration=clip_max_duration,
            video_duration=metadata.duration,
            # intro/outro zones only make sense for the whole video, not a chosen section or live recording
            skip_edges=(metadata.offset == 0 and range_end is None and not live_minutes),
            theme=theme,
        )
        t_sc = time.perf_counter()

        # Stage 5: Reframing 9:16 + Captions + Final Export
        update_status(job_id, "Fetching clip video & rendering 9:16 shorts" if audio_first
                      else "Reframing to 9:16 & burning animated captions", 85)
        clip_source = None
        if audio_first:
            from pipeline.fast_url import ClipVideoFetcher
            clip_source = ClipVideoFetcher(source, job_dir, metadata.offset, job_dir / "audio.wav").get
        clips = await export_all_clips(
            ranked_clips=ranked,
            source_video=metadata.source_path,
            transcript=transcript,
            job_dir=job_dir,
            output_dir=output_dir,
            captions_enabled=captions,
            caption_preset=caption_preset,
            layout_info=(await layout_task) if layout_task else {"layout": "blurred_fit", "confidence": 0.0},
            caption_font=caption_font,
            title_overlay=bool(title_overlay) and config.TITLE_OVERLAY,
            clip_source=clip_source,
        )
        # Times in the original video (useful when only a range was processed)
        for c in clips:
            c["source_start"] = round(c["start"] + metadata.offset, 2)
            c["source_end"] = round(c["end"] + metadata.offset, 2)
        import json as _json
        with open(output_dir / "clips.json", "w", encoding="utf-8") as f:
            _json.dump(clips, f, indent=2, ensure_ascii=False)
        t_ex = time.perf_counter()
        logger.info(
            f"[{job_id}] Timings: download {t_dl - t0:.1f}s | transcribe {t_tr - t_dl:.1f}s | "
            f"analyze {t_an - t_tr:.1f}s | score {t_sc - t_an:.1f}s | export {t_ex - t_sc:.1f}s | "
            f"total {t_ex - t0:.1f}s"
        )

        update_status(job_id, "Completed", 100, status="completed", clips=clips)

        # Webhook callback if requested
        callback_url = webhook_url or config.WEBHOOK_URL
        if callback_url:
            try:
                import httpx
                headers = {"x-webhook-secret": config.WEBHOOK_SECRET}
                async with httpx.AsyncClient() as client:
                    await client.post(callback_url, json={"job_id": job_id, "status": "completed", "clips": clips}, headers=headers, timeout=10.0)
                logger.info(f"Webhook notification sent to {callback_url}")
            except Exception as e:
                logger.warning(f"Webhook notification failed: {e}")

        return {"job_id": job_id, "status": "completed", "clips": clips}

    except Exception as e:
        logger.exception(f"Pipeline failed for job {job_id}: {e}")
        update_status(job_id, "Failed", 0, status="failed", error=str(e))
        return {"job_id": job_id, "status": "failed", "error": str(e)}

# ── Modal Function Registrations ──
if app is not None:
    @app.function(image=image, secrets=secrets, gpu="any", timeout=1800)
    def modal_process_video(job_payload: dict):
        return asyncio.run(run_pipeline(**job_payload))

    @app.function(image=image)
    @modal.web_endpoint(method="POST")
    def submit_job(req: JobRequest):
        modal_process_video.spawn(req.model_dump())
        return {"job_id": req.job_id, "status": "queued"}

    @app.function(image=image)
    @modal.web_endpoint(method="GET")
    def get_status(job_id: str):
        return _job_statuses.get(job_id, {"status": "not_found"})

# ── Local CLI Runner ──
if __name__ == "__main__":
    import argparse
    import uuid

    parser = argparse.ArgumentParser(description="AI Viral Clip Generator")
    parser.add_argument("source", help="YouTube/Twitch URL or local video path")
    parser.add_argument("--clips", type=int, default=8, help="Number of clips to generate (min 2)")
    parser.add_argument("--min", type=float, default=15.0, help="Min clip duration (seconds)")
    parser.add_argument("--max", type=float, default=30.0, help="Max clip duration (seconds, up to 180)")
    parser.add_argument("--start", type=float, default=None, help="Only use the video from this time (seconds)")
    parser.add_argument("--end", type=float, default=None, help="Only use the video up to this time (seconds)")
    parser.add_argument("--live-minutes", type=float, default=None,
                        help="For a live stream: how many minutes to record before clipping (default 10)")
    parser.add_argument("--font", type=str, default=None, help="Caption font override, e.g. Impact")
    parser.add_argument("--hindi-quality", choices=["fast", "best"], default="fast",
                        help="Hindi captions: fast (turbo) or best (large-v3, ~4x slower transcription)")
    parser.add_argument("--no-captions", action="store_true", help="Disable burned-in captions")
    from pipeline.caption import CAPTION_STYLES, STYLE_ALIASES
    parser.add_argument("--preset", "--style", dest="preset", type=str, default=config.DEFAULT_CAPTION_STYLE,
                        choices=sorted(list(CAPTION_STYLES) + list(STYLE_ALIASES)), help="Caption style")
    parser.add_argument("--no-title", action="store_true", help="Don't burn the clip title into the video")
    parser.add_argument("--job-id", type=str, default=None, help="Explicit job ID")
    parser.add_argument("--theme", choices=["all", "fighting", "comedy", "romance"], default="all",
                        help="Thematic focus: all, fighting (action), comedy (funny), or romance (drama)")
    parser.add_argument("--language", type=str, default=config.TRANSCRIBE_LANGUAGE,
                        help="Spoken language: 'auto' or ISO code (hi, en, mr, bn, ta, te, ...). Captions keep that language's script.")

    args = parser.parse_args()
    if args.no_title:
        config.TITLE_OVERLAY = False
    # Hindi titles/emoji must not crash printing on a cp1252 Windows console or log file
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    job_id = args.job_id or str(uuid.uuid4())[:8]
    
    print(f"🚀 Starting Viral Clipper (Job: {job_id}, Preset: {args.preset}) on {args.source}")
    res = asyncio.run(run_pipeline(
        job_id=job_id,
        source=args.source,
        clip_count=args.clips,
        clip_min_duration=args.min,
        clip_max_duration=args.max,
        captions=not args.no_captions,
        caption_preset=args.preset,
        language=args.language,
        range_start=args.start,
        range_end=args.end,
        live_minutes=args.live_minutes,
        caption_font=args.font,
        title_overlay=not args.no_title,
        hindi_quality=args.hindi_quality,
        theme=args.theme,
    ))

    if res["status"] == "completed":
        print(f"\n🎉 Successfully generated {len(res['clips'])} viral clips!")
        for c in res["clips"]:
            print(f"  • [{c['score']}/100] {c['title']} ({c['duration']}s) -> {c['path']}")
    else:
        print(f"\n❌ Job failed: {res.get('error')}")
