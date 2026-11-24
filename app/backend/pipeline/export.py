"""Optimized final clip export with single-pass rendering, concurrent encoding, and manifest generation."""
import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import config
from .score import RankedClip
from .transcribe import TranscriptionResult
from .reframe import reframe_clip, get_video_layout, _video_encoder_args, _detect_clip_layout
from .caption import generate_captions

logger = logging.getLogger("viral_clipper.export")

async def export_clip(
    video_path: Path,
    subtitle_path: Optional[Path],
    output_path: Path,
    layout_info: Optional[Dict[str, Any]] = None
) -> Tuple[Path, str]:
    """Single clip export wrapper with single-pass reframing and optional caption burn."""
    return await reframe_clip(
        video_path=video_path,
        start=0.0,
        end=0.0,
        output_path=output_path,
        layout_info=layout_info,
        subtitle_path=subtitle_path
    )

async def export_all_clips(
    ranked_clips: List[RankedClip],
    source_video: Path,
    transcript: TranscriptionResult,
    job_dir: Path,
    output_dir: Path,
    captions_enabled: bool = True,
    caption_preset: str = "streamer",
    max_concurrency: Optional[int] = None,
    caption_font: Optional[str] = None,
    title_overlay: bool = True,
    clip_source=None,  # optional async (idx, start, end) -> (video_path, shift) for per-clip video
    layout_info: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Optimized OpenShorts export pipeline:
    1. Pre-detect master video layout (split_cam, single_speaker, blurred_fit) ONCE.
    2. Generate non-overlapping animated karaoke captions using selected preset (hormozi, lime, pill, clean).
    3. Render reframing + caption burning + audio loudnorm in a SINGLE FFmpeg pass per clip.
    4. Run up to 2 clips concurrently to fully saturate multi-core CPUs/GPUs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Video master layout detection (cached)
    if layout_info is None:
        layout_info = await asyncio.to_thread(get_video_layout, source_video)
    _video_encoder_args()  # probe NVENC once, before the parallel workers start
    max_concurrency = max_concurrency or config.EXPORT_CONCURRENCY
    layout_type = layout_info.get("layout", "blurred_fit")
    logger.info(f"Video master layout identified: '{layout_type}' (caption preset: '{caption_preset}')")

    # 2. Gather all words for subtitle slicing
    all_words = []
    for s in transcript.segments:
        all_words.extend(s.words)

    # 3. Concurrent export worker pool
    sem = asyncio.Semaphore(max_concurrency)

    async def _process_single_clip(i: int, clip: RankedClip) -> Dict[str, Any]:
        # Fetch this clip's video right away (all clips download in parallel); only the
        # rendering below is limited to `max_concurrency` at a time.
        prefetched = None
        if clip_source is not None:
            prefetched = await clip_source(i, clip.start, clip.end)
        async with sem:
            clip_prefix = f"clip_{i:02d}_score{clip.score}"
            final_target = output_dir / f"{clip_prefix}.mp4"
            sub_target = job_dir / f"{clip_prefix}.ass"

            # Where this clip's video comes from: the whole source, or (audio-first mode for
            # links) a small section downloaded just for this clip, starting `shift` s into the job.
            clip_video, shift = prefetched if prefetched else (source_video, 0.0)
            v_start, v_end = clip.start - shift, clip.end - shift

            # Find the webcam inside THIS clip (compilations / moving overlays put it elsewhere)
            clip_layout = layout_info
            try:
                # Trust the clip's own result: a wrong webcam crop looks worse than a full-frame fit
                clip_layout = await asyncio.to_thread(_detect_clip_layout, clip_video, v_start, v_end)
            except Exception as e:
                logger.warning(f"Per-clip layout detection failed ({e}); using video layout.")
            clip_layout_type = clip_layout.get("layout", "blurred_fit")

            # Captions + title label share one ASS file burned in the same ffmpeg pass
            subtitle_file: Optional[Path] = None
            has_words = any(w.end >= clip.start and w.start <= clip.end for w in all_words)
            want_title = title_overlay and bool(clip.title)
            if (captions_enabled and has_words) or want_title:
                await generate_captions(
                    all_words,
                    clip.start,
                    clip.end,
                    sub_target,
                    layout_type=clip_layout_type,
                    preset=caption_preset,
                    title=clip.title if want_title else None,
                    include_words=captions_enabled,
                    font=caption_font,
                )
                subtitle_file = sub_target

            # Execute single-pass reframing + caption burn
            logger.info(f"[{i}/{len(ranked_clips)}] Exporting (single-pass): {clip.title} ({clip.duration:.1f}s)...")
            await reframe_clip(
                video_path=clip_video,
                start=v_start,
                end=v_end,
                output_path=final_target,
                layout_info=clip_layout,
                subtitle_path=subtitle_file
            )

            return {
                "id": f"clip_{i:02d}",
                "filename": final_target.name,
                "path": str(final_target),
                "url": f"/output/{final_target.name}",
                "score": clip.score,
                "layout": clip_layout_type,
                "title": clip.title,
                "hook_line": clip.hook_line,
                "start": clip.start,
                "end": clip.end,
                "duration": clip.duration,
                "viral_factors": clip.viral_factors,
                "reason": clip.reason,
                "scene_type": getattr(clip, "scene_type", "general"),
            }

    tasks = [_process_single_clip(i, clip) for i, clip in enumerate(ranked_clips, start=1)]
    results = await asyncio.gather(*tasks)

    # Ensure results preserve clip ordering
    results.sort(key=lambda x: x["id"])

    # Save manifest
    manifest_path = output_dir / "clips.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"All {len(results)} clips exported successfully in single-pass. Manifest: {manifest_path}")
    return results
