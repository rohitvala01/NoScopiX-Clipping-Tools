"""Multimodal signal analysis for identifying high-engagement viral candidate windows."""
import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import config
from .download import VideoMetadata
from .transcribe import TranscriptionResult

logger = logging.getLogger("viral_clipper.analyze")

@dataclass
class SignalWindow:
    start: float
    end: float
    scores: Dict[str, float]
    combined_score: float

@dataclass
class AnalysisResult:
    windows: List[SignalWindow]
    candidates: List[SignalWindow]

def _load_audio_data(audio_path: Path) -> Tuple[Optional[np.ndarray], int]:
    """Load audio data once efficiently using soundfile (fast) or librosa fallback."""
    if not audio_path.exists():
        return None, 16000
    try:
        import soundfile as sf
        data, sr = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        return data, sr
    except Exception:
        try:
            import librosa
            data, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            return data, sr
        except Exception as e:
            logger.warning(f"Failed to load audio for signal analysis: {e}")
            return None, 16000

def _normalize(arr: np.ndarray) -> np.ndarray:
    return (arr - np.min(arr)) / (np.ptp(arr) + 1e-6)


def _analyze_audio_energy(y: Optional[np.ndarray], sr: int, duration: float, window_size: float = 5.0) -> List[Tuple[float, float]]:
    """Compute audio RMS energy over 5s windows with a 1s hop (numpy only, no librosa)."""
    if y is None or len(y) == 0:
        return []
    try:
        # Windowed mean-square via cumulative sum: O(n) time, no big frame matrix
        win = int(sr * window_size)
        csum = np.concatenate([[0.0], np.cumsum(y.astype(np.float64) ** 2)])
        centers = np.arange(0, len(y), sr)
        lo = np.clip(centers - win // 2, 0, len(y))
        hi = np.clip(centers + win // 2, 0, len(y))
        rms = np.sqrt((csum[hi] - csum[lo]) / np.maximum(hi - lo, 1))
        norm = _normalize(rms)
        return [(float(i), float(v)) for i, v in enumerate(norm) if i <= duration]
    except Exception as e:
        logger.warning(f"Audio energy analysis failed: {e}")
        return []

def _detect_laughter_applause(y: Optional[np.ndarray], sr: int, duration: float) -> List[Tuple[float, float]]:
    """Heuristic detector for laughter/applause/hype: loud, noisy, high-frequency bursts.
    Scores each 1s frame by high-band (2-7 kHz) energy share x spectral flatness x loudness."""
    if y is None or len(y) == 0:
        return []
    try:
        n_fft = 2048
        hann = np.hanning(n_fft).astype(np.float32)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        band = (freqs >= 2000) & (freqs <= 7000)
        hs, fl, ld = [], [], []
        chunk = 600  # process 10 minutes of 1s frames at a time to bound memory
        n_frames = int(np.ceil(len(y) / sr))
        for c0 in range(0, n_frames, chunk):
            seg = y[c0 * sr:(c0 + chunk) * sr]
            k = len(seg) // sr
            if k == 0:
                break
            frames = seg[: k * sr].reshape(k, sr)
            # Average spectrum over short sub-windows of each 1s frame
            sub = frames[:, : (frames.shape[1] // n_fft) * n_fft].reshape(frames.shape[0], -1, n_fft)
            spec = np.abs(np.fft.rfft(sub * hann, axis=2)).mean(axis=1) + 1e-9
            hs.append(spec[:, band].sum(axis=1) / spec.sum(axis=1))
            fl.append(np.exp(np.mean(np.log(spec), axis=1)) / np.mean(spec, axis=1))
            ld.append(np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)))
        high_share, flatness, loud = np.concatenate(hs), np.concatenate(fl), np.concatenate(ld)
        score = _normalize(high_share) * 0.4 + _normalize(flatness) * 0.3 + _normalize(loud) * 0.3
        return [(float(i), float(v)) for i, v in enumerate(score) if i <= duration]
    except Exception as e:
        logger.warning(f"Laughter/applause heuristic failed: {e}")
        return []

def _parse_heatmap(heatmap_data: Optional[List[Dict[str, float]]], duration: float) -> List[Tuple[float, float]]:
    """Normalize YouTube 'Most Replayed' heatmap data to timestamp points."""
    if not heatmap_data:
        return []
    results = []
    for item in heatmap_data:
        t = (item["start"] + item["end"]) / 2.0
        val = max(0.0, min(1.0, item["value"]))
        results.append((t, val))
    return results

def _analyze_chat_spikes(chat_replay: Optional[List[Dict[str, Any]]], duration: float) -> List[Tuple[float, float]]:
    """Count chat velocity (msgs/10s) and boost for hype emotes."""
    if not chat_replay or duration <= 0:
        return []
    
    bucket_size = 10.0
    num_buckets = int(math.ceil(duration / bucket_size))
    buckets = [0.0] * num_buckets
    hype_keywords = {"lul", "kekw", "lol", "pog", "pogchamp", "omg", "wtffff", "lmao", "wtf"}

    for msg in chat_replay:
        t = msg.get("time_sec", 0.0)
        text = msg.get("text", "").lower()
        idx = int(t // bucket_size)
        if 0 <= idx < num_buckets:
            weight = 1.0
            words = set(text.split())
            if words.intersection(hype_keywords):
                weight += 2.0
            if text.isupper() and len(text) > 4:
                weight += 1.5
            buckets[idx] += weight

    arr = np.array(buckets)
    if np.ptp(arr) == 0:
        return []
    norm = (arr - np.min(arr)) / np.ptp(arr)
    
    results = []
    for i, val in enumerate(norm):
        t = i * bucket_size + (bucket_size / 2.0)
        results.append((t, float(val)))
    return results

# Thematic keywords for extracting high-retention scenes (Action/Fighting, Comedy/Humor, Romantic/Drama)
FIGHT_ACTION_KEYWORDS = {
    # English
    "kill", "killed", "killing", "shoot", "shooting", "shot", "gun", "guns", "fight", "fighting",
    "war", "attack", "attacking", "headshot", "clutch", "punch", "die", "died", "dead", "destroy",
    "destroyed", "damage", "police", "chase", "enemy", "enemies", "danger", "blast", "bomb", "hit",
    "smash", "brawl", "action", "sniper", "weapon", "weapons", "defeat", "execute", "revenge",
    "blood", "rage", "screaming", "rush", "flank", "target", "combat", "showdown", "knock", "knocked",
    "bullet", "bullets", "grenade", "bazooka", "assault", "ambush", "battle", "survive", "deadly",
    # Hindi / Hinglish
    "maar", "maaro", "maara", "maardiya", "marna", "marenge", "thok", "thoko", "thokna", "goli",
    "goliyan", "dushman", "khatam", "bhaag", "bhaago", "bachao", "bhedna", "ladai", "lafda", "jhagda",
    "pitai", "dhoya", "tod", "toda", "phod", "phoda", "udaa", "udao", "udade", "pakad", "gira", "giraya",
    "loot", "kutta", "chodunga", "maarunga", "daalunga", "bachke", "aag", "kabza", "gang", "gangster",
    "shikaar", "phodna", "phat", "panga", "hathiyar", "tode", "dhoyenge", "girega", "bachna",
}

COMEDY_HUMOR_KEYWORDS = {
    # English
    "lol", "lmao", "rofl", "haha", "hahaha", "funny", "hilarious", "joke", "jokes", "prank", "pranks",
    "troll", "trolls", "trolling", "fail", "fails", "stupid", "crazy", "insane", "wtf", "ridiculous",
    "roast", "roasting", "cracked", "comedy", "meme", "memes", "clown", "embarrassing", "noob", "bot",
    "laugh", "laughing", "laughs", "funniest", "awkward", "oops", "silly", "unreal",
    # Hindi / Hinglish
    "hasi", "has", "haso", "hasna", "hasenge", "mazak", "mazaak", "pagal", "chutiya", "bewakoof",
    "gadha", "chirkut", "bakwaas", "popat", "hadd", "ulti", "sharam", "joker", "maza", "khulle",
    "arre", "bhains", "paani", "oyee", "waah", "dhanya", "scene", "phati", "fatt", "halkat",
    "bawasir", "beizzati", "nautanki", "drama", "ullu", "chomu", "locha", "ghapla", "jhamela",
    "chutiye", "kya kar raha", "gayi", "bhai ye kya", "has mat", "epic fail", "kamina",
}

ROMANTIC_DRAMA_KEYWORDS = {
    # English
    "love", "romance", "romantic", "flirt", "flirting", "cute", "crush", "dating", "date", "marry",
    "marriage", "propose", "proposal", "girlfriend", "boyfriend", "gf", "bf", "kiss", "kissing",
    "hug", "hugging", "sweetheart", "darling", "baby", "babe", "beautiful", "handsome", "gorgeous",
    "heart", "heartbreak", "crying", "tears", "miss", "emotional", "breakup", "feelings", "soulmate",
    "sweet", "lover", "attractive", "hot", "relationship", "couple", "together", "forever",
    # Hindi / Hinglish
    "pyaar", "pyar", "ishq", "mohabbat", "dil", "jaan", "shona", "babu", "sundar", "khubsoorat",
    "khoobsurat", "shaadi", "shadi", "biah", "biwi", "patni", "patana", "patayi", "chumma", "pappi",
    "yaad", "rona", "ro mat", "aansoo", "dhokha", "kash", "apna", "deewani", "deewana", "aashiq",
    "prem", "preet", "meethi", "mehbooba", "sanam", "jaaneman", "pyari", "dilwale", "mohabbatein",
}


def _analyze_scene_themes(
    transcript: TranscriptionResult,
    duration: float,
    target_theme: str = "all",
    window_sec: float = 5.0
) -> List[Tuple[float, float]]:
    """
    Score timeline windows based on high-interest themes:
    - Fighting / Action / Combat
    - Comedy / Humor / Funny fails
    - Romantic / Flirting / Emotional Drama
    Filters out boring random driving, walking, or idle chatter.
    """
    import re as _re
    if not transcript or not transcript.segments:
        return []

    num_windows = int(math.ceil(duration / window_sec))
    scores = [0.0] * num_windows

    for seg in transcript.segments:
        text = seg.text.lower()
        tokens = set(_re.findall(r"\b\w+\b", text))

        fight_hits = len(tokens.intersection(FIGHT_ACTION_KEYWORDS))
        comedy_hits = len(tokens.intersection(COMEDY_HUMOR_KEYWORDS))
        romance_hits = len(tokens.intersection(ROMANTIC_DRAMA_KEYWORDS))

        exclamation_boost = min(1.5, text.count("!") * 0.4)
        question_boost = min(1.0, text.count("?") * 0.25)
        upper_boost = 0.5 if any(w.isupper() and len(w) > 3 for w in seg.text.split()) else 0.0

        if target_theme == "fighting":
            thematic = fight_hits * 2.0 + exclamation_boost + upper_boost
        elif target_theme == "comedy":
            thematic = comedy_hits * 2.0 + exclamation_boost
        elif target_theme == "romance":
            thematic = romance_hits * 2.0 + question_boost
        else:  # "all": highlight whichever exciting theme is present
            max_theme = max(fight_hits, comedy_hits, romance_hits)
            sum_theme = fight_hits + comedy_hits + romance_hits
            thematic = (max_theme * 1.5) + (sum_theme * 0.4) + exclamation_boost + question_boost + upper_boost

        if thematic > 0:
            idx_start = max(0, int((seg.start - 2.0) // window_sec))
            idx_end = min(num_windows, int(math.ceil((seg.end + 2.0) / window_sec)))
            for idx in range(idx_start, idx_end):
                scores[idx] = max(scores[idx], thematic)

    arr = np.array(scores)
    if np.ptp(arr) == 0:
        return [(float(i * window_sec + window_sec / 2.0), 0.5) for i in range(num_windows)]
    norm = (arr - np.min(arr)) / np.ptp(arr)
    return [(float(i * window_sec + window_sec / 2.0), float(norm[i])) for i in range(num_windows)]


def _analyze_speech_patterns(transcript: TranscriptionResult, duration: float) -> List[Tuple[float, float]]:
    """Detect dynamic speech rate shifts (rapid punchlines or dramatic pauses)."""
    if not transcript.segments:
        return []
    
    window_sec = 5.0
    num_windows = int(math.ceil(duration / window_sec))
    rates = [0.0] * num_windows

    for seg in transcript.segments:
        seg_dur = max(0.2, seg.end - seg.start)
        num_words = len(seg.words) or len(seg.text.split())
        wpm = (num_words / seg_dur) * 60.0
        
        idx = int(((seg.start + seg.end) / 2.0) // window_sec)
        if 0 <= idx < num_windows:
            rates[idx] = max(rates[idx], wpm)

    arr = np.array(rates)
    if np.ptp(arr) == 0:
        return []
    norm = (arr - np.min(arr)) / np.ptp(arr)
    
    results = []
    for i, val in enumerate(norm):
        t = i * window_sec + (window_sec / 2.0)
        results.append((t, float(val)))
    return results

def detect_scene_changes(video_path: Path) -> List[float]:
    """Find scene cut timestamps with ffmpeg's scene score on a tiny, 5 fps, GPU-decoded stream.
    Far faster than PySceneDetect on full-resolution frames."""
    import re as _re
    base = ["ffmpeg", "-hide_banner", "-nostats"]
    tail = ["-an", "-sn", "-f", "null", "-"]
    # GPU: decode, drop to 5 fps and shrink on the GPU; only tiny frames reach the CPU
    gpu_cmd = base + ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", str(video_path),
                      "-vf", "fps=5,scale_cuda=160:90:format=nv12,hwdownload,format=nv12,"
                             "select='gt(scene,0.30)',showinfo"] + tail
    cpu_cmd = base + ["-hwaccel", "auto", "-i", str(video_path),
                      "-vf", "fps=5,scale=160:-2,select='gt(scene,0.30)',showinfo"] + tail
    try:
        from .gpu import gpu_decode_ok
        res = None
        if gpu_decode_ok(video_path):
            res = subprocess.run(gpu_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res is None or res.returncode != 0:
            res = subprocess.run(cpu_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode != 0:
            logger.warning(f"Scene detection failed: {res.stderr[-300:]}")
            return []
        times = [float(m) for m in _re.findall(r"pts_time:([0-9.]+)", res.stderr)]
        logger.info(f"Scene detection: {len(times)} cuts")
        return times
    except Exception as e:
        logger.warning(f"Scene detection fallback: {e}")
        return []

def _interpolate_signal(signal_points: List[Tuple[float, float]], target_time: float) -> float:
    """Find nearest or interpolated value for a timestamp."""
    if not signal_points:
        return 0.0
    times = [p[0] for p in signal_points]
    values = [p[1] for p in signal_points]
    return float(np.interp(target_time, times, values, left=0.0, right=0.0))

def _merge_candidates(
    windows: List[SignalWindow],
    min_duration: float,
    max_duration: float,
    min_gap: float,
    target_count: int
) -> List[SignalWindow]:
    """Rank high-energy windows, expand to target duration, and remove overlaps."""
    # Sort windows by score descending
    sorted_windows = sorted(windows, key=lambda w: w.combined_score, reverse=True)
    
    selected: List[SignalWindow] = []
    for w in sorted_windows:
        cand_start = max(0.0, w.start - 2.0)
        cand_end = cand_start + max_duration
        
        # Check collision with already selected
        overlap = False
        for s in selected:
            if not (cand_end + min_gap < s.start or cand_start - min_gap > s.end):
                overlap = True
                break
        
        if not overlap:
            selected.append(SignalWindow(
                start=round(cand_start, 2),
                end=round(cand_end, 2),
                scores=w.scores,
                combined_score=round(w.combined_score, 3)
            ))
        
        if len(selected) >= target_count:
            break
            
    # Sort chronologically
    selected.sort(key=lambda x: x.start)
    return selected

async def analyze_signals(
    video_path: Path,
    transcript: TranscriptionResult,
    metadata: VideoMetadata,
    job_dir: Path,
    scene_changes: Optional[List[float]] = None,
    min_duration: float = config.DEFAULT_CLIP_MIN_DURATION,
    max_duration: float = config.DEFAULT_CLIP_MAX_DURATION,
    target_count: int = config.CANDIDATE_COUNT,
    theme: str = "all",
) -> AnalysisResult:
    """Run multimodal and thematic scene signals to generate top candidate windows."""
    duration = metadata.duration
    audio_path = job_dir / "audio.wav"
    
    logger.info(f"Extracting audio energy, reaction signals, and '{theme}' thematic scenes...")
    audio_data, audio_sr = _load_audio_data(audio_path)
    audio_energy = _analyze_audio_energy(audio_data, audio_sr, duration)
    laughter = _detect_laughter_applause(audio_data, audio_sr, duration)
    heatmap = _parse_heatmap(metadata.heatmap_data, duration)
    chat_spikes = _analyze_chat_spikes(metadata.chat_replay, duration)
    speech_patterns = _analyze_speech_patterns(transcript, duration)
    scene_themes = _analyze_scene_themes(transcript, duration, target_theme=theme)
    if scene_changes is None:
        scene_changes = detect_scene_changes(video_path)

    # Determine dynamic active weights
    active_weights = dict(config.SIGNAL_WEIGHTS)
    if not heatmap:
        active_weights.pop("most_replayed", None)
    if not chat_spikes:
        active_weights.pop("chat_spikes", None)
    if not scene_changes:
        active_weights.pop("scene_changes", None)
    if not scene_themes:
        active_weights.pop("scene_thematic", None)

    # Re-normalize weights to sum to 1.0
    total_w = sum(active_weights.values())
    active_weights = {k: v / total_w for k, v in active_weights.items()}
    logger.info(f"Active signal weights: {active_weights}")

    # Build 5-second evaluation grid
    step = 2.5  # 50% overlap for granular detection
    num_steps = max(1, int(duration // step))
    all_windows: List[SignalWindow] = []

    for i in range(num_steps):
        t_center = i * step + (step / 2.0)
        if t_center > duration:
            break

        scores = {}
        if "scene_thematic" in active_weights:
            scores["scene_thematic"] = _interpolate_signal(scene_themes, t_center)
        if "audio_energy" in active_weights:
            scores["audio_energy"] = _interpolate_signal(audio_energy, t_center)
        if "laughter_applause" in active_weights:
            scores["laughter_applause"] = _interpolate_signal(laughter, t_center)
        if "most_replayed" in active_weights:
            scores["most_replayed"] = _interpolate_signal(heatmap, t_center)
        if "chat_spikes" in active_weights:
            scores["chat_spikes"] = _interpolate_signal(chat_spikes, t_center)
        if "speech_patterns" in active_weights:
            scores["speech_patterns"] = _interpolate_signal(speech_patterns, t_center)
        if "scene_changes" in active_weights:
            # Score higher if close to a scene change
            nearest_scene_dist = min([abs(t_center - sc) for sc in scene_changes], default=999.0)
            scores["scene_changes"] = 1.0 if nearest_scene_dist < 2.0 else 0.0

        combined = sum(scores.get(k, 0.0) * w for k, w in active_weights.items())
        all_windows.append(SignalWindow(
            start=round(max(0.0, t_center - 2.5), 2),
            end=round(min(duration, t_center + 2.5), 2),
            scores=scores,
            combined_score=round(combined, 3)
        ))

    candidates = _merge_candidates(
        all_windows,
        min_duration=min_duration,
        max_duration=max_duration,
        min_gap=config.MIN_GAP_BETWEEN_CLIPS,
        target_count=target_count
    )

    logger.info(f"Generated {len(candidates)} high-potential candidates for LLM judging")
    return AnalysisResult(windows=all_windows, candidates=candidates)
