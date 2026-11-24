"""Configuration for the viral-clipper backend."""
import os
from pathlib import Path

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load .env if present ──
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k_str = k.strip()
                    v_str = v.strip().strip('"').strip("'")
                    if k_str:
                        os.environ.setdefault(k_str, v_str)  # real env vars win over .env
    except Exception:
        pass

# ── Video Constraints ──
MAX_VIDEO_DURATION = int(os.getenv("MAX_VIDEO_DURATION", "7200"))  # 2 hours max
SUPPORTED_FORMATS = ["mp4", "mov", "webm", "mkv"]

# ── Transcription ──
# "auto" -> large-v3-turbo on a CUDA GPU (fast + good Hindi), "small" on CPU
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "auto")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # "auto", "cpu", "cuda"
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "default")
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))     # 1 = greedy (much faster, ~same quality)
WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "8"))   # batched GPU inference
# Model for Hindi / Indian languages (more accurate than turbo on Hindi; "" = use WHISPER_MODEL)
WHISPER_MODEL_INDIC = os.getenv("WHISPER_MODEL_INDIC", "large-v3")
WHISPER_INDIC_COMPUTE = os.getenv("WHISPER_INDIC_COMPUTE", "int8_float16")  # fits 6 GB GPUs

# Spoken language: "auto" to detect, or an ISO code like "hi", "en", "mr", "bn", "ta".
TRANSCRIBE_LANGUAGE = os.getenv("TRANSCRIBE_LANGUAGE", "auto")
# Caption script: "native" keeps each language's own script (Hindi -> Devanagari),
# "roman" converts Hindi to Latin-script Hinglish.
CAPTION_SCRIPT = os.getenv("CAPTION_SCRIPT", "roman")
# Experimental: steer Hindi transcripts to keep English words in Latin letters
# ("ये game बहुत अच्छा है"). Can cause repetition loops on noisy audio.
CODE_MIXED_PROMPT = os.getenv("CODE_MIXED_PROMPT", "0") == "1"

# ── Analysis & Signal Windows ──
WINDOW_SIZE_SEC = 5.0
CANDIDATE_COUNT = 30

SIGNAL_WEIGHTS = {
    "scene_thematic": 0.30,  # fighting, comedy, romance / high-entertainment narrative
    "audio_energy": 0.15,
    "laughter_applause": 0.20,
    "most_replayed": 0.20,
    "chat_spikes": 0.10,
    "speech_patterns": 0.05,
}
SIGNAL_LLM_RATIO = (0.4, 0.6)  # 40% signal score, 60% LLM score

# ── Clipping ──
DEFAULT_CLIP_MIN_DURATION = 5.0
DEFAULT_CLIP_MAX_DURATION = 25.0
ABSOLUTE_MIN_CLIP_DURATION = 1.0
ABSOLUTE_MAX_CLIP_DURATION = 180.0  # YouTube Shorts / Reels allow up to 3 minutes
MIN_CLIPS = 2
DEFAULT_CLIP_COUNT = 10
MIN_GAP_BETWEEN_CLIPS = 8.0  # seconds
# Quality filters
MIN_CLIP_SCORE = int(os.getenv("MIN_CLIP_SCORE", "40"))      # drop clips scoring below this (best 2 always kept)
INTRO_SKIP_SEC = float(os.getenv("INTRO_SKIP_SEC", "15"))   # videos >= 3 min: ignore clips starting this early
OUTRO_SKIP_SEC = float(os.getenv("OUTRO_SKIP_SEC", "20"))   # ...or ending this close to the end

# ── Reframing ──
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
FACE_SMOOTHING_ALPHA = 0.15  # EMA smoothing for face tracking

# ── Encoding ──
# "auto" uses NVIDIA NVENC when available, else libx264. Or force "libx264" / "h264_nvenc".
VIDEO_ENCODER = os.getenv("VIDEO_ENCODER", "auto")
EXPORT_CONCURRENCY = int(os.getenv("EXPORT_CONCURRENCY", "3"))
# Links: download audio first and only the video sections the clips need (much faster)
AUDIO_FIRST = os.getenv("AUDIO_FIRST", "1") == "1"
# GPU video decoding for export: "auto" (GPU when possible, CPU fallback) or "none"
VIDEO_HWACCEL = os.getenv("VIDEO_HWACCEL", "auto")

# ── Captions (ASS format) ──
CAPTION_LATIN_FONT = os.getenv("CAPTION_LATIN_FONT", "Arial Black")
# Caption size multiplier (1.0 = default ~84px on a 1080x1920 frame)
CAPTION_SIZE_SCALE = float(os.getenv("CAPTION_SIZE_SCALE", "1.0"))
# Vertical CENTRE of the caption line, as a fraction of frame height (0 = top, 0.5 = middle, 1 = bottom)
CAPTION_Y_SPLIT = float(os.getenv("CAPTION_Y_SPLIT", "0.48"))      # split-cam: just below the cam/game seam
CAPTION_Y_DEFAULT = float(os.getenv("CAPTION_Y_DEFAULT", "0.55"))  # other layouts: around the middle
# Split-cam layout: webcam band on top takes this share of the height, gameplay the rest
SPLIT_CAM_TOP_FRACTION = float(os.getenv("SPLIT_CAM_TOP_FRACTION", "0.40"))
# Burn the clip title into the short (black label box, like streamer clip channels)
TITLE_OVERLAY = os.getenv("TITLE_OVERLAY", "1") == "1"
DEFAULT_CAPTION_STYLE = os.getenv("DEFAULT_CAPTION_STYLE", "streamer")
# Optional font override for captions ("" = use the style's own font)
CAPTION_FONT_OVERRIDE = os.getenv("CAPTION_FONT", "")
# Average uppercase glyph width (em) per font, used to keep caption lines inside the frame
CAPTION_FONT_WIDTHS = {
    "Arial Black": 0.80, "Impact": 0.56, "Segoe UI Black": 0.74, "Comic Sans MS": 0.68,
    "Bahnschrift": 0.62, "Franklin Gothic Medium": 0.60, "Verdana": 0.74, "Segoe UI": 0.62,
    "Tahoma": 0.66,
}
# Fonts for non-Latin scripts (word-level override so English words keep the Latin font)
CAPTION_SCRIPT_FONTS = {
    "devanagari": os.getenv("CAPTION_DEVANAGARI_FONT", "Nirmala UI" if os.name == "nt" else "Noto Sans Devanagari"),
    "bengali": "Nirmala UI" if os.name == "nt" else "Noto Sans Bengali",
    "gurmukhi": "Nirmala UI" if os.name == "nt" else "Noto Sans Gurmukhi",
    "gujarati": "Nirmala UI" if os.name == "nt" else "Noto Sans Gujarati",
    "tamil": "Nirmala UI" if os.name == "nt" else "Noto Sans Tamil",
    "telugu": "Nirmala UI" if os.name == "nt" else "Noto Sans Telugu",
    "kannada": "Nirmala UI" if os.name == "nt" else "Noto Sans Kannada",
    "malayalam": "Nirmala UI" if os.name == "nt" else "Noto Sans Malayalam",
    "arabic": "Segoe UI" if os.name == "nt" else "Noto Sans Arabic",
    "cjk": "Microsoft YaHei" if os.name == "nt" else "Noto Sans CJK SC",
}
CAPTION_FONT = "Montserrat-Bold"
CAPTION_FONT_SIZE = 48
CAPTION_PRIMARY_COLOR = "&H00FFFFFF"     # White (ASS &HAABBGGRR)
CAPTION_HIGHLIGHT_COLOR = "&H0000FFFF"   # Yellow
CAPTION_OUTLINE_COLOR = "&H00000000"     # Black
CAPTION_OUTLINE_WIDTH = 3
CAPTION_SHADOW_DEPTH = 2
CAPTION_WORDS_PER_CHUNK = 6
CAPTION_MARGIN_BOTTOM = 80

# ── LLM Settings ──
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-flash-latest")
LLM_MAX_RETRIES = 2
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "40"))
QUOTA_COOLDOWN_SEC = int(os.getenv("QUOTA_COOLDOWN_SEC", "600"))  # skip a quota-blocked Gemini model this long
LLM_CONTEXT_PADDING = 10.0  # seconds on each side

# ── Cloud Storage (R2 / S3) ──
R2_ENDPOINT = os.getenv("R2_ENDPOINT", "")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY", "")
R2_BUCKET = os.getenv("R2_BUCKET", "viral-clipper")
R2_PUBLIC_URL_PREFIX = os.getenv("R2_PUBLIC_URL_PREFIX", "")

# ── Webhooks ──
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
