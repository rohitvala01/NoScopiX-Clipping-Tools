"""Video download and metadata retrieval pipeline using yt-dlp."""
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
import config

logger = logging.getLogger("viral_clipper.download")

@dataclass
class VideoMetadata:
    source_path: Path
    title: str
    duration: float
    url: Optional[str] = None
    heatmap_data: Optional[List[Dict[str, float]]] = None
    chat_replay: Optional[List[Dict[str, Any]]] = None
    offset: float = 0.0  # where the fetched part starts in the original video (seconds)

def _get_video_duration(path: Path) -> float:
    """Use ffprobe to determine duration in seconds."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception as e:
        logger.warning(f"Failed to probe duration for {path}: {e}")
        return 0.0

def _extract_heatmap_data(info_json_path: Path) -> Optional[List[Dict[str, float]]]:
    """Parse YouTube 'Most Replayed' heatmap data if present."""
    if not info_json_path.exists():
        return None
    try:
        with open(info_json_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        heatmap = info.get("heatmap")
        if heatmap and isinstance(heatmap, list):
            # heatmap items typically have 'start_time', 'end_time', 'value' (0..1)
            formatted = []
            for item in heatmap:
                formatted.append({
                    "start": float(item.get("start_time", 0.0)),
                    "end": float(item.get("end_time", 0.0)),
                    "value": float(item.get("value", 0.0))
                })
            return formatted
    except Exception as e:
        logger.warning(f"Error parsing heatmap: {e}")
    return None

def _extract_chat_replay(chat_json_path: Path) -> Optional[List[Dict[str, Any]]]:
    """Parse chat replay messages if downloaded."""
    if not chat_json_path.exists():
        return None
    messages = []
    try:
        with open(chat_json_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # Extract timestamp (in seconds or millis) and message text
                    item = data.get("replayChatItemAction", {})
                    actions = item.get("actions", [])
                    for act in actions:
                        renderer = act.get("addChatItemAction", {}).get("item", {}).get("liveChatTextMessageRenderer", {})
                        if renderer:
                            time_usec = int(renderer.get("timestampUsec", 0))
                            msg = "".join([r.get("text", "") for r in renderer.get("message", {}).get("runs", [])])
                            messages.append({"time_sec": time_usec / 1_000_000.0, "text": msg})
                except Exception:
                    continue
    except Exception as e:
        logger.warning(f"Error parsing chat: {e}")
    return messages if messages else None

def _ytdlp(for_download: bool = True) -> List[str]:
    """yt-dlp base command. YouTube throttles downloads unless yt-dlp can run its JavaScript
    challenge, so point it at Node.js (already installed for the web app) when downloading.
    Just reading video info is ~0.5 s faster without it."""
    ytdlp_bin = shutil.which("yt-dlp") or str(Path(sys.executable).parent / "yt-dlp")
    if Path(ytdlp_bin).exists():
        cmd = [ytdlp_bin, "--no-warnings"]
    else:
        cmd = [sys.executable, "-m", "yt_dlp", "--no-warnings"]
    if for_download and shutil.which("node"):
        cmd += ["--js-runtimes", "node"]
    return cmd


# Prefer H.264 (the GPU can decode it; AV1 forces slow CPU decoding on RTX 20-series)
_YT_FORMAT = ("bestvideo[ext=mp4][vcodec^=avc1][height<=1080]+bestaudio[ext=m4a]/"
              "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best")


def _is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


# Just-ended live streams that YouTube hasn't processed yet: served as tens of thousands of
# 2-second pieces, many refused (HTTP 401), at ~50 KB/s. Not practical to download.
PROCESSING_MSG = ("This live stream ended recently and YouTube is still processing it. Until YouTube "
                  "finishes (for very long streams this can take hours or more), it downloads extremely "
                  "slowly or fails. Please try again later, or use a saved copy of the video.")


def _yt_info(url: str) -> Dict[str, Any]:
    # Ask only for the fields we use: the full info of a long stream can be hundreds of MB.
    fields = "id,title,duration,is_live,live_status,was_live,thumbnail,channel,uploader,heatmap"
    cmd = _ytdlp(for_download=False) + ["--no-playlist", "--skip-download", "-O", f"%(.{{{fields}}})j", url]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    if res.returncode != 0:
        raise RuntimeError(f"Could not read video info: {res.stderr.strip()[-300:]}")
    return json.loads(res.stdout.strip().splitlines()[-1])


def probe_source(source: str) -> Dict[str, Any]:
    """Quick look at a URL or local file before processing: title, length, live status."""
    source = source.strip().strip('"')
    if not _is_url(source):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {source}")
        return {"source": str(p), "type": "file", "title": p.stem, "duration": _get_video_duration(p),
                "is_live": False, "thumbnail": None}
    info = _yt_info(source)
    is_live = bool(info.get("is_live")) or info.get("live_status") == "is_live"
    return {
        "source": source,
        "type": "url",
        "title": info.get("title") or "Video",
        "duration": None if is_live else float(info.get("duration") or 0.0),
        "is_live": is_live,
        "live_status": info.get("live_status"),
        "processing": info.get("live_status") == "post_live",
        "thumbnail": info.get("thumbnail"),
        "channel": info.get("channel") or info.get("uploader"),
        "has_heatmap": bool(info.get("heatmap")),
    }


def _ffmpeg_cut(src: Path, dst: Path, start: float, end: float) -> None:
    """Fast stream-copy cut of [start, end] (snaps to the nearest keyframes)."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.2f}", "-i", str(src), "-t", f"{end - start:.2f}",
           "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", "-avoid_negative_ts", "make_zero", str(dst)]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0 or not dst.exists():
        raise RuntimeError(f"Could not cut the selected range: {res.stderr[-300:]}")


def _record_live(url: str, job_dir: Path, minutes: float) -> Path:
    """Record the next `minutes` of a live stream (YouTube / Twitch / any HLS live)."""
    res = subprocess.run(
        _ytdlp() + ["-g", "-f", "bestvideo[height<=1080][protocol*=m3u8]+bestaudio[protocol*=m3u8]/best[height<=1080][protocol*=m3u8]/best[height<=1080]/best", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    stream_urls = [u for u in res.stdout.splitlines() if u.startswith("http")]
    if res.returncode != 0 or not stream_urls:
        raise RuntimeError(f"Could not open the live stream: {res.stderr.strip()[-300:]}")
    out = job_dir / "source.mp4"
    secs = max(30, int(minutes * 60))
    logger.info(f"Recording {secs // 60} min of the live stream...")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", stream_urls[0]]
    if len(stream_urls) > 1:  # separate audio stream
        cmd += ["-i", stream_urls[1], "-map", "0:v:0", "-map", "1:a:0"]
    cmd += ["-t", str(secs), "-c", "copy", "-bsf:a", "aac_adtstoasc", str(out)]
    rec = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"Live recording failed: {rec.stderr[-300:]}")
    return out


async def download_video(
    source: str,
    job_dir: Path,
    start: Optional[float] = None,
    end: Optional[float] = None,
    live_minutes: Optional[float] = None,
) -> VideoMetadata:
    """Fetch the source (URL, live stream, or local file), optionally only the [start, end] range.
    Returned timestamps are relative to the fetched part; `offset` maps them back."""
    import asyncio
    job_dir.mkdir(parents=True, exist_ok=True)
    source = source.strip().strip('"')
    has_range = start is not None and end is not None and end > start
    offset = float(start) if has_range else 0.0

    # ── Local file ──
    if not _is_url(source):
        local_path = Path(source)
        if not local_path.exists():
            raise FileNotFoundError(f"Local video file not found: {source}")
        if has_range:
            target = job_dir / f"source{local_path.suffix or '.mp4'}"
            await asyncio.to_thread(_ffmpeg_cut, local_path, target, float(start), float(end))
        else:
            target = local_path  # use in place: no multi-GB copy
        meta = VideoMetadata(source_path=target, title=local_path.stem,
                             duration=_get_video_duration(target), url=None)
        meta.offset = offset
        return meta

    # ── Live stream: record the next N minutes ──
    info: Dict[str, Any] = {}
    try:
        info = await asyncio.to_thread(_yt_info, source)
    except Exception as e:
        logger.warning(f"Could not pre-read video info ({e}); downloading anyway.")
    if info.get("is_live") or info.get("live_status") == "is_live":
        path = await asyncio.to_thread(_record_live, source, job_dir, live_minutes or 10)
        meta = VideoMetadata(source_path=path, title=info.get("title", "Live stream"),
                             duration=_get_video_duration(path), url=source)
        meta.offset = 0.0
        return meta

    # ── Normal video / finished stream (optionally just a section) ──
    output_template = str(job_dir / "source.%(ext)s")
    cmd = _ytdlp() + [
        "--format", _YT_FORMAT,
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--concurrent-fragments", "8",
        "--no-playlist",
        "--max-filesize", "4G",
        "--output", output_template,
    ]
    # Section download (ffmpeg over HTTP) is slow; for videos under ~30 min it's faster to
    # grab the whole file and cut locally.
    full_len = float(info.get("duration") or 0)
    section_dl = has_range and (full_len == 0 or full_len > 1800)
    if section_dl:
        cmd += ["--download-sections", f"*{float(start):.2f}-{float(end):.2f}"]
    cmd.append(source)

    logger.info(f"Running yt-dlp on {source}" + (f" (section {start:.0f}-{end:.0f}s)" if has_range else "") + "...")
    res = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace")
    if res.returncode != 0:
        logger.error(f"yt-dlp error: {res.stderr}")
        raise RuntimeError(f"yt-dlp failed to download video: {res.stderr.strip()[-300:]}")

    video_path = job_dir / "source.mp4"
    if not video_path.exists():
        video_candidates = [c for c in job_dir.glob("source.*") if c.suffix.lower() in [".mp4", ".mkv", ".webm"]]
        if not video_candidates:
            raise FileNotFoundError(f"Downloaded video not found in {job_dir}")
        video_path = video_candidates[0]

    info_json = job_dir / "source.info.json"
    title = info.get("title") or "Video"
    if info_json.exists():
        try:
            with open(info_json, "r", encoding="utf-8") as f:
                title = json.load(f).get("title", title)
        except Exception:
            pass
    if has_range and not section_dl:
        cut = job_dir / f"section{video_path.suffix}"
        await asyncio.to_thread(_ffmpeg_cut, video_path, cut, float(start), float(end))
        video_path = cut
    duration = _get_video_duration(video_path)

    heatmap = _extract_heatmap_data(info_json)
    if heatmap and has_range:  # shift "Most Replayed" data into the section's timeline
        heatmap = [
            {"start": h["start"] - offset, "end": h["end"] - offset, "value": h["value"]}
            for h in heatmap if h["end"] > offset and h["start"] < float(end)
        ] or None
    chat = _extract_chat_replay(job_dir / "source.live_chat.json")

    logger.info(f"Downloaded '{title}' (Duration: {duration:.1f}s, Heatmap: {heatmap is not None}, Chat: {chat is not None})")
    meta = VideoMetadata(source_path=video_path, title=title, duration=duration, url=source,
                         heatmap_data=heatmap, chat_replay=chat)
    meta.offset = offset
    return meta
