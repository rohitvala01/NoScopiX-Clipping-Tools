"""Smart 9:16 Portrait reframing with Webcam Split-Screen, Face Tracking, and Blurred-Fit."""
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import config

logger = logging.getLogger("viral_clipper.reframe")

_ENCODER_CACHE: Dict[str, List[str]] = {}


def split_cam_height() -> int:
    """Height of the webcam band at the top of split-cam shorts (even number)."""
    return (int(config.OUTPUT_HEIGHT * config.SPLIT_CAM_TOP_FRACTION) // 2) * 2


def _video_encoder_args() -> List[str]:
    """Pick the fastest working H.264 encoder: NVIDIA NVENC if usable, else libx264."""
    if "args" in _ENCODER_CACHE:
        return _ENCODER_CACHE["args"]
    x264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22"]
    nvenc = ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", "22", "-b:v", "0"]
    choice = config.VIDEO_ENCODER
    args = x264
    if choice in ("auto", "h264_nvenc"):
        test = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
             "color=black:s=256x256:d=0.2", *nvenc, "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if test.returncode == 0:
            args = nvenc
        else:
            logger.info("NVENC unavailable, using libx264.")
    logger.info(f"Video encoder: {args[1]}")
    _ENCODER_CACHE["args"] = args
    return args


def _hwaccel_args() -> List[str]:
    """Decode the source on the GPU when possible (NVDEC / D3D11); ffmpeg falls back to the
    CPU by itself for codecs the GPU can't decode (e.g. AV1 on RTX 20-series)."""
    return ["-hwaccel", config.VIDEO_HWACCEL] if config.VIDEO_HWACCEL not in ("", "none") else []


PRESCALE_HEIGHT = 1080
# 4K and other >1080p sources: decode AND shrink to 1080p on the GPU, then hand small frames to
# the CPU filters (crop / blur / captions). About 2x faster than resizing 4K frames on the CPU.
_GPU_PRESCALE_INPUT = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
_GPU_PRESCALE_CHAIN = f"scale_cuda=-2:{PRESCALE_HEIGHT}:format=nv12,hwdownload,format=nv12,"
_GPU_PRESCALE_GRAPH = f"[0:v]scale_cuda=-2:{PRESCALE_HEIGHT}:format=nv12,hwdownload,format=nv12[src];"
_DIMS_CACHE: Dict[str, Tuple[int, int]] = {}


def _source_dims(path: Path) -> Tuple[int, int]:
    key = str(path)
    if key not in _DIMS_CACHE:
        res = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=width,height", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True)
        try:
            w, h = (int(x) for x in res.stdout.strip().split(",")[:2])
        except Exception:
            w, h = 1920, 1080
        _DIMS_CACHE[key] = (w, h)
    return _DIMS_CACHE[key]


def _use_gpu_prescale(path: Path) -> bool:
    from .gpu import gpu_decode_ok
    return _source_dims(path)[1] > PRESCALE_HEIGHT and gpu_decode_ok(path)


def _strip_hwaccel(cmd: List[str]) -> List[str]:
    out, skip = [], False
    for c in cmd:
        if skip:
            skip = False
            continue
        if c == "-hwaccel":
            skip = True
            continue
        out.append(c)
    return out


def _encode_args() -> List[str]:
    return [
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        *_video_encoder_args(),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
    ]


# Face detection: YuNet (neural, accurate on small webcam insets) with Haar cascade fallback
_face_cascade = None
if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
    try:
        _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception:
        _face_cascade = None
_YUNET_PATH = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"


def _find_faces(frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
    """Return faces as (x, y, w, h, score) in frame pixels."""
    fh_, fw_ = frame.shape[:2]
    if _YUNET_PATH.exists() and hasattr(cv2, "FaceDetectorYN"):
        try:
            det = cv2.FaceDetectorYN.create(str(_YUNET_PATH), "", (fw_, fh_), 0.75, 0.3, 50)
            _, faces = det.detect(frame)
            if faces is None:
                return []
            return [(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14])) for f in faces]
        except Exception as e:
            logger.debug(f"YuNet failed ({e}); using Haar cascade")
    if _face_cascade:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = _face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                               minSize=(max(36, fh_ // 28),) * 2)
        return [(float(x), float(y), float(w), float(h), 0.6) for (x, y, w, h) in found]
    return []


def _grab_frames(path: Path, start: float, end: float, count: int, src_w: int, src_h: int, scale: float):
    """Decode `count` evenly spaced, downscaled BGR frames from [start, end] with one ffmpeg call."""
    d = max(0.5, end - start)
    out_w = max(2, int(src_w * scale) // 2 * 2)
    out_h = max(2, int(src_h * scale) // 2 * 2)
    tail = ["-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    seek = ["-ss", f"{start:.2f}", "-t", f"{d:.2f}"]
    # GPU: decode + drop frames + shrink on the GPU, download only the few small frames we need
    gpu_cmd = ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", *seek,
               "-i", str(path), "-an", "-sn",
               "-vf", f"fps={count / d:.4f},scale_cuda={out_w}:{out_h}:format=nv12,hwdownload,format=nv12",
               *tail]
    cpu_cmd = ["ffmpeg", "-v", "error", *seek, "-i", str(path), "-an", "-sn",
               "-vf", f"fps={count / d:.4f},scale={out_w}:{out_h}", *tail]
    from .gpu import gpu_decode_ok
    res = None
    if gpu_decode_ok(path):
        res = subprocess.run(gpu_cmd, capture_output=True)
    if res is None or res.returncode != 0 or not res.stdout:
        res = subprocess.run(cpu_cmd, capture_output=True)
    size = out_w * out_h * 3
    for k in range(len(res.stdout) // size):
        yield np.frombuffer(res.stdout[k * size:(k + 1) * size], dtype=np.uint8).reshape(out_h, out_w, 3)


def _detect_clip_layout(clip_path: Path, start: Optional[float] = None, end: Optional[float] = None) -> Dict[str, Any]:
    """
    Analyze frames to determine visual structure:
    - 'split_cam': a streamer webcam (a face that stays in one place) + gameplay.
    - 'single_speaker': one large subject filling the frame (vlog / podcast).
    - 'blurred_fit': no stable face.
    With start/end, samples only that time range of the file (per-clip detection).
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return {"layout": "blurred_fit", "confidence": 0.0}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    det_scale = min(1.0, 960.0 / max(w, 1))  # detect on <=960px-wide frames for speed

    sample_count = 14
    if start is not None and end is not None and end > start:
        d = end - start
        positions = [("ms", t * 1000.0) for t in np.linspace(start + 0.05 * d, end - 0.05 * d, sample_count)]
    else:
        positions = [("frame", int(i)) for i in np.linspace(total_frames * 0.1, total_frames * 0.9, sample_count)]

    def _frames():
        if positions and positions[0][0] == "ms":
            # One sequential ffmpeg decode of the range is far faster than random seeks
            yield from _grab_frames(clip_path, start, end, sample_count, w, h, det_scale)
            return
        for _, pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, fr = cap.read()
            if ret:
                yield cv2.resize(fr, None, fx=det_scale, fy=det_scale, interpolation=cv2.INTER_AREA) if det_scale < 1.0 else fr

    faces_detected = []
    valid_frames = 0
    for frame in _frames():
        if float(frame.mean()) < 12:  # black / flash frames carry no layout info
            continue
        valid_frames += 1
        for (fx, fy, fw, fh, score) in _find_faces(frame):
            fx, fy, fw, fh = (v / det_scale for v in (fx, fy, fw, fh))
            faces_detected.append({
                "cx": (fx + fw / 2.0) / w, "cy": (fy + fh / 2.0) / h,
                "rel_area": (fw * fh) / (w * h), "w": fw, "h": fh, "score": score,
            })
    cap.release()

    if not faces_detected or valid_frames == 0:
        return {"layout": "blurred_fit", "confidence": 0.0}

    # Group faces by position: a webcam face stays put, game characters move around.
    clusters: List[List[Dict[str, Any]]] = []
    for f in faces_detected:
        for c in clusters:
            if abs(c[0]["cx"] - f["cx"]) < 0.08 and abs(c[0]["cy"] - f["cy"]) < 0.10:
                c.append(f)
                break
        else:
            clusters.append([f])
    best = max(clusters, key=lambda c: (len(c), sum(x["score"] for x in c)))
    hits = len(best)
    stable = hits >= max(2, int(0.3 * valid_frames))
    if not stable:
        return {"layout": "blurred_fit", "confidence": hits / max(1, valid_frames)}

    cx = float(np.median([f["cx"] for f in best]))
    cy = float(np.median([f["cy"] for f in best]))
    fw = float(np.median([f["w"] for f in best]))
    fh = float(np.median([f["h"] for f in best]))
    area = float(np.median([f["rel_area"] for f in best]))
    conf = hits / valid_frames

    if area >= 0.035:
        # Face fills a big part of the frame: talking-head / vlog
        return {"layout": "single_speaker", "mean_x": cx, "confidence": conf}

    logger.info(f"Webcam face at X={cx:.2f}, Y={cy:.2f} ({hits}/{valid_frames} frames) -> split_cam")
    return {
        "layout": "split_cam",
        "src_size": (w, h),
        "webcam_center": (cx, cy),
        "webcam_size": (fw, fh),
        "confidence": conf,
    }

# Global layout cache to prevent repetitive probing for clips from the same source
_LAYOUT_CACHE: Dict[str, Dict[str, Any]] = {}

def get_video_layout(video_path: Path) -> Dict[str, Any]:
    """
    Detect visual layout (split_cam, single_speaker, blurred_fit) once for the video.
    Results are cached so subsequent clips probe in 0ms.
    """
    key = str(video_path.resolve())
    if key in _LAYOUT_CACHE:
        return _LAYOUT_CACHE[key]

    import hashlib, tempfile
    probe_target = Path(tempfile.gettempdir()) / f"vc_layout_probe_{hashlib.md5(key.encode()).hexdigest()[:10]}{video_path.suffix or '.mp4'}"
    try:
        # Extract 4-second probe at 15% through video (avoids intro black screens)
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", "10.0",
            "-i", str(video_path),
            "-t", "4.0",
            "-c", "copy",
            str(probe_target)
        ], capture_output=True)
        if not probe_target.exists() or probe_target.stat().st_size == 0:
            # Short video: probe from the start instead
            subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-t", "4.0", "-c", "copy",
                            str(probe_target)], capture_output=True)

        if probe_target.exists():
            layout_info = _detect_clip_layout(probe_target)
            probe_target.unlink(missing_ok=True)
        else:
            layout_info = {"layout": "blurred_fit", "confidence": 0.0}
    except Exception as e:
        logger.warning(f"Error during video layout probe: {e}")
        layout_info = {"layout": "blurred_fit", "confidence": 0.0}

    _LAYOUT_CACHE[key] = layout_info
    logger.info(f"Master video layout: {layout_info.get('layout')} (confidence: {layout_info.get('confidence', 0):.2f})")
    return layout_info

def _escape_sub_path(path: Path) -> str:
    """Format subtitle path for FFmpeg libass filter."""
    return str(path).replace("\\", "/").replace(":", "\\:")

def _build_split_cam_command(
    source_video: Path,
    start: float,
    end: float,
    webcam_info: Dict[str, Any],
    output_path: Path,
    subtitle_path: Optional[Path] = None,
    prescale: bool = False,
) -> List[str]:
    """
    Builds the streamer-shorts stacked 9:16 layout in a SINGLE PASS:
    - Top (1080 x cam_h, ~40%): zoomed crop on the streamer's webcam (face first).
    - Bottom (1080 x rest, ~60%): centered gameplay.
    - Seam: thin dark divider; the title label sits just above it, captions just below it.
    """
    target_w = config.OUTPUT_WIDTH   # 1080
    target_h = config.OUTPUT_HEIGHT  # 1920
    cam_h = split_cam_height()       # 768 by default
    game_h = target_h - cam_h        # 1152
    duration = max(1.0, end - start)

    cx, cy = webcam_info["webcam_center"]
    fw, fh = webcam_info["webcam_size"]
    src_w, src_h = webcam_info.get("src_size", (1920, 1080))
    if prescale:  # frames arrive already shrunk to 1080p on the GPU
        f = PRESCALE_HEIGHT / src_h
        src_w, src_h = (int(src_w * f) // 2) * 2, PRESCALE_HEIGHT
        fw, fh = fw * f, fh * f

    # Ensure all dimensions and offsets are EVEN numbers for H.264
    # ~2.6 face-widths: frames the head and shoulders while staying inside the webcam inset
    cam_box_w = (int(max(360, fw * 2.6)) // 2) * 2
    cam_box_h = (int(cam_box_w * (cam_h / target_w)) // 2) * 2

    cam_box_w = min(cam_box_w, (src_w // 2) * 2)
    cam_box_h = min(cam_box_h, (src_h // 2) * 2)
    px_cx = int(cx * src_w)
    px_cy = int(cy * src_h)

    cam_crop_x = (max(0, min(src_w - cam_box_w, px_cx - (cam_box_w // 2))) // 2) * 2
    # Face slightly above centre, leaving room for the title label at the bottom of the cam
    cam_crop_y = (max(0, min(src_h - cam_box_h, px_cy - int(cam_box_h * 0.42))) // 2) * 2

    filter_complex = (
        (_GPU_PRESCALE_GRAPH if prescale else "[0:v]null[src];") +
        f"[src]split=2[v_cam][v_game];"
        f"[v_cam]crop={cam_box_w}:{cam_box_h}:{cam_crop_x}:{cam_crop_y},scale={target_w}:{cam_h},setsar=1[top_cam];"
        f"[v_game]scale={target_w}:{game_h}:force_original_aspect_ratio=increase,"
        f"crop={target_w}:{game_h},setsar=1[bottom_game];"
        f"[top_cam][bottom_game]vstack=inputs=2[v_stacked];"
        f"[v_stacked]drawbox=x=0:y={cam_h - 2}:w={target_w}:h=4:color=black@0.9:t=fill[v_boxed]"
    )

    if subtitle_path and subtitle_path.exists():
        sub_str = _escape_sub_path(subtitle_path)
        filter_complex += f";[v_boxed]ass='{sub_str}'[outv]"
    else:
        filter_complex += f";[v_boxed]null[outv]"

    return [
        "ffmpeg", "-y",
        *(_GPU_PRESCALE_INPUT if prescale else _hwaccel_args()),
        "-ss", str(start),
        "-i", str(source_video),
        "-t", str(duration),
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "0:a?",
        *_encode_args(),
        str(output_path)
    ]

def _build_single_speaker_command(
    source_video: Path,
    start: float,
    end: float,
    mean_x: float,
    output_path: Path,
    subtitle_path: Optional[Path] = None,
    prescale: bool = False,
) -> List[str]:
    """Single full-height 9:16 portrait crop tracking the speaker with single-pass caption burn."""
    target_w = config.OUTPUT_WIDTH
    target_h = config.OUTPUT_HEIGHT
    duration = max(1.0, end - start)

    filter_str = (_GPU_PRESCALE_CHAIN if prescale else "") + (
        f"crop='min(iw, ih*9/16)':ih:'min(max(0, iw*{mean_x:.3f} - (ih*9/32)), iw - min(iw, ih*9/16))':0,"
        f"scale={target_w}:{target_h}"
    )

    if subtitle_path and subtitle_path.exists():
        sub_str = _escape_sub_path(subtitle_path)
        filter_str += f",ass='{sub_str}'"

    return [
        "ffmpeg", "-y",
        *(_GPU_PRESCALE_INPUT if prescale else _hwaccel_args()),
        "-ss", str(start),
        "-i", str(source_video),
        "-t", str(duration),
        "-vf", filter_str,
        *_encode_args(),
        str(output_path)
    ]

def _build_blur_fallback_command(
    source_video: Path,
    start: float,
    end: float,
    output_path: Path,
    subtitle_path: Optional[Path] = None,
    prescale: bool = False,
) -> List[str]:
    """Center crop with stylish blurred background and single-pass caption burn."""
    target_w = config.OUTPUT_WIDTH
    target_h = config.OUTPUT_HEIGHT
    duration = max(1.0, end - start)

    filter_complex = (
        (_GPU_PRESCALE_GRAPH if prescale else "[0:v]null[src];") +
        f"[src]split=2[bg][fg];"
        f"[bg]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},boxblur=20:5[bg_blur];"
        f"[fg]scale={target_w}:-1:force_original_aspect_ratio=decrease[fg_scaled];"
        f"[bg_blur][fg_scaled]overlay=(W-w)/2:(H-h)/2[v_overlay]"
    )

    if subtitle_path and subtitle_path.exists():
        sub_str = _escape_sub_path(subtitle_path)
        filter_complex += f";[v_overlay]ass='{sub_str}'[outv]"
    else:
        filter_complex += f";[v_overlay]null[outv]"

    return [
        "ffmpeg", "-y",
        *(_GPU_PRESCALE_INPUT if prescale else _hwaccel_args()),
        "-ss", str(start),
        "-i", str(source_video),
        "-t", str(duration),
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "0:a?",
        *_encode_args(),
        str(output_path)
    ]

async def reframe_clip(
    video_path: Path,
    start: float,
    end: float,
    output_path: Path,
    layout_info: Optional[Dict[str, Any]] = None,
    subtitle_path: Optional[Path] = None
) -> Tuple[Path, str]:
    """
    Cut, reframe to 9:16, and burn animated captions in a SINGLE pass.
    Returns (output_path, layout_type).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if layout_info is None:
        layout_info = get_video_layout(video_path)

    layout = layout_info.get("layout", "blurred_fit")

    def build(prescale: bool) -> List[str]:
        if layout == "split_cam":
            return _build_split_cam_command(video_path, start, end, layout_info, output_path, subtitle_path, prescale)
        if layout == "single_speaker":
            return _build_single_speaker_command(video_path, start, end, layout_info.get("mean_x", 0.5),
                                                 output_path, subtitle_path, prescale)
        return _build_blur_fallback_command(video_path, start, end, output_path, subtitle_path, prescale)

    prescale = _use_gpu_prescale(video_path)
    cmd = build(prescale)

    def _run(c: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(c, capture_output=True, text=True, encoding="utf-8", errors="replace")

    # Run ffmpeg in a worker thread so several clips really encode in parallel.
    res = await asyncio.to_thread(_run, cmd)
    if res.returncode != 0 and prescale:
        logger.warning("GPU resize failed for this clip; retrying without it.")
        cmd = build(False)
        res = await asyncio.to_thread(_run, cmd)
    if res.returncode != 0 and "-hwaccel" in cmd:
        logger.warning("GPU decode failed for this clip; retrying with CPU decode.")
        res = await asyncio.to_thread(_run, _strip_hwaccel(cmd))
    if res.returncode != 0:
        logger.warning(f"Single-pass layout ({layout}) failed ({res.stderr[-400:]}). Falling back to blur fit.")
        fallback_cmd = _build_blur_fallback_command(video_path, start, end, output_path, subtitle_path)
        res = await asyncio.to_thread(_run, _strip_hwaccel(fallback_cmd))
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg export failed: {res.stderr[-400:]}")
        layout = "blurred_fit"

    return output_path, layout
