"""Scoring, sentence snapping, and final ranking pipeline."""
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import config
from .analyze import SignalWindow
from .transcribe import TranscriptionResult
try:
    from llm import judge_candidates, ClipJudgment
except ImportError:
    from ..llm import judge_candidates, ClipJudgment

logger = logging.getLogger("viral_clipper.score")

@dataclass
class RankedClip:
    start: float
    end: float
    duration: float
    title: str
    hook_line: str
    score: int
    signal_score: float
    llm_score: int
    viral_factors: List[str]
    reason: str
    scene_type: str = "general"

def _extract_context(
    start: float,
    end: float,
    transcript: TranscriptionResult,
    padding: float = config.LLM_CONTEXT_PADDING
) -> Dict[str, str]:
    """Slice transcript into clip text, context before, and context after."""
    before_texts = []
    clip_texts = []
    after_texts = []

    for seg in transcript.segments:
        if seg.end < start - padding:
            continue
        elif seg.end <= start:
            before_texts.append(seg.text)
        elif seg.start >= end + padding:
            break
        elif seg.start >= end:
            after_texts.append(seg.text)
        else:
            clip_texts.append(seg.text)

    return {
        "text": " ".join(clip_texts),
        "context_before": " ".join(before_texts[-2:]),
        "context_after": " ".join(after_texts[:2]),
    }

def _snap_to_sentence(
    start: float,
    end: float,
    transcript: TranscriptionResult,
    min_duration: float,
    max_duration: float
) -> Tuple[float, float]:
    """Snap start/end timestamps to nearest sentence or word boundary to never cut mid-word."""
    all_words = []
    for seg in transcript.segments:
        all_words.extend(seg.words)

    if not all_words:
        return start, end

    # Find closest word starts
    best_start = start
    for w in all_words:
        if abs(w.start - start) < 1.0:
            best_start = w.start
            break

    # Find closest word end with clean punctuation
    best_end = end
    for w in reversed(all_words):
        if w.end <= best_start + max_duration and w.end >= best_start + min_duration:
            best_end = w.end
            # If word ends with punctuation, strongly prefer it
            if w.word.endswith((".", "!", "?", ",", "।", "॥")):
                best_end = w.end
                break

    duration = best_end - best_start
    if duration > max_duration:
        best_end = best_start + max_duration
    elif duration < min_duration:
        best_end = best_start + min_duration

    return round(best_start, 2), round(best_end, 2)

# Hinglish / English phrases that mark intros, outros and channel promos
_FILLER_PHRASES = (
    "agli video", "next video", "milte hain", "thanks for watching", "thank you for watching",
    "subscribe", "bell icon", "like kar", "like karo", "comment kar", "comment karo", "playlist",
    "end tak", "aaj ki video", "aaj ke video", "is video mein", "iss video mein", "welcome back",
    "share kar", "link description", "description mein",
)


def _filler_reason(llm_flag: bool, text: str, start: float, end: float, video_duration: float,
                   skip_edges: bool = True) -> str:
    if llm_flag:
        return "intro/outro/promo per AI"
    t = text.lower()
    if sum(p in t for p in _FILLER_PHRASES) >= 1 and len(t.split()) < 40:
        hit = next(p for p in _FILLER_PHRASES if p in t)
        return f"promo phrase '{hit}'"
    if skip_edges and video_duration >= 180:  # full long videos: intro / outro zones
        if start < config.INTRO_SKIP_SEC:
            return f"in first {config.INTRO_SKIP_SEC:.0f}s (intro)"
        if end > video_duration - config.OUTRO_SKIP_SEC:
            return f"in last {config.OUTRO_SKIP_SEC:.0f}s (outro)"
    return ""


async def score_and_rank(
    candidates: List[SignalWindow],
    transcript: TranscriptionResult,
    clip_count: int = config.DEFAULT_CLIP_COUNT,
    min_duration: float = config.DEFAULT_CLIP_MIN_DURATION,
    max_duration: float = config.DEFAULT_CLIP_MAX_DURATION,
    video_duration: float = 0.0,
    skip_edges: bool = True,
    theme: str = "all",
) -> List[RankedClip]:
    """Evaluate candidates with Gemini LLM, snap boundaries, and produce ranked clips."""
    logger.info(f"Preparing {len(candidates)} candidates for LLM scoring (theme: '{theme}')...")
    
    prepared_candidates = []
    for cand in candidates:
        ctx = _extract_context(cand.start, cand.end, transcript)
        prepared_candidates.append({
            "start": cand.start,
            "end": cand.end,
            "text": ctx["text"],
            "context_before": ctx["context_before"],
            "context_after": ctx["context_after"],
            "signal_score": cand.combined_score
        })

    # Run LLM judgments
    judgments = await judge_candidates(
        prepared_candidates,
        min_duration=min_duration,
        max_duration=max_duration,
        language=transcript.language,
        theme=theme,
    )

    ranked_clips: List[RankedClip] = []
    sig_weight, llm_weight = config.SIGNAL_LLM_RATIO
    all_words = [w for seg in transcript.segments for w in seg.words]
    scored: List[RankedClip] = []
    reserve: List[RankedClip] = []  # filler-flagged clips, used only to reach MIN_CLIPS

    for j in judgments:
        # Snap to clean sentence / word boundaries
        snapped_start, snapped_end = _snap_to_sentence(
            j.start, j.end, transcript, min_duration, max_duration
        )
        if video_duration > 0:
            snapped_end = min(snapped_end, round(video_duration, 2))
            snapped_start = max(0.0, min(snapped_start, snapped_end - min_duration))
        duration = round(snapped_end - snapped_start, 2)

        # Find corresponding signal score
        cand_match = next((c for c in candidates if abs(c.start - j.start) < 4.0), None)
        sig_score = cand_match.combined_score if cand_match else 0.5

        # Calculate thematic boost
        st = getattr(j, "scene_type", "general")
        theme_boost = 0
        if theme == "fighting":
            theme_boost = 15 if st == "fighting_action" else -15
        elif theme == "comedy":
            theme_boost = 15 if st == "comedy_humor" else -15
        elif theme == "romance":
            theme_boost = 15 if st == "romantic_drama" else -15
        elif theme == "all":
            # Any recognized thematic scene gets a boost over generic
            if st in ("fighting_action", "comedy_humor", "romantic_drama"):
                theme_boost = 8

        final_score = int(round(
            (sig_score * 100.0 * sig_weight) + (j.score * llm_weight) + theme_boost
        ))
        # Shorts with almost no talking rarely work (and get no captions): penalize them
        n_words = sum(1 for w in all_words if w.end >= snapped_start and w.start <= snapped_end)
        words_per_sec = n_words / max(1.0, duration)
        if words_per_sec < 0.5:
            final_score = int(final_score * (0.5 + words_per_sec))

        # Drop intros, outros and channel promos: they never work as standalone shorts
        clip_text = " ".join(w.word for w in all_words if w.end >= snapped_start and w.start <= snapped_end)
        filler_reason = _filler_reason(j.is_filler, clip_text, snapped_start, snapped_end, video_duration, skip_edges)
        target = reserve if filler_reason else scored
        if filler_reason:
            logger.info(f"Skipping {snapped_start:.0f}-{snapped_end:.0f}s ({filler_reason}): {j.title}")

        vf = list(j.viral_factors)
        if st != "general" and st not in vf:
            vf.insert(0, st)

        target.append(RankedClip(
            start=snapped_start,
            end=snapped_end,
            duration=duration,
            title=j.title,
            hook_line=j.hook_line,
            score=min(100, max(1, final_score)),
            signal_score=round(sig_score, 2),
            llm_score=j.score,
            viral_factors=vf,
            reason=j.reason,
            scene_type=st,
        ))

    # Pick the best-scoring clips first, skipping any that overlap an already chosen one.
    # Clips below MIN_CLIP_SCORE are dropped rather than padded in (the best MIN_CLIPS are always kept).
    scored.sort(key=lambda c: c.score, reverse=True)
    for clip in scored:
        if len(ranked_clips) >= config.MIN_CLIPS and clip.score < config.MIN_CLIP_SCORE:
            logger.info(f"Skipping weak clip ({clip.score} < {config.MIN_CLIP_SCORE}): {clip.title}")
            continue
        overlap = any(
            not (clip.end + config.MIN_GAP_BETWEEN_CLIPS < rc.start or clip.start - config.MIN_GAP_BETWEEN_CLIPS > rc.end)
            for rc in ranked_clips
        )
        if not overlap:
            ranked_clips.append(clip)
        if len(ranked_clips) >= clip_count:
            break
    # Never return fewer than MIN_CLIPS: top up with the best skipped clips if needed
    reserve.sort(key=lambda c: c.score, reverse=True)
    for clip in reserve:
        if len(ranked_clips) >= min(config.MIN_CLIPS, clip_count):
            break
        if not any(not (clip.end + config.MIN_GAP_BETWEEN_CLIPS < rc.start or
                        clip.start - config.MIN_GAP_BETWEEN_CLIPS > rc.end) for rc in ranked_clips):
            logger.info(f"Using skipped clip to reach the minimum of {config.MIN_CLIPS}: {clip.title}")
            ranked_clips.append(clip)
    ranked_clips.sort(key=lambda c: c.score, reverse=True)

    if len(ranked_clips) < clip_count:
        logger.info(f"Only {len(ranked_clips)} of {clip_count} requested clips were strong enough to keep")
    logger.info(f"Selected top {len(ranked_clips)} ranked viral clips")
    return ranked_clips
