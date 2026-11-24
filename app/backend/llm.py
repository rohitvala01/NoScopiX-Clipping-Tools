"""LLM interface for scoring candidate viral clips with Google Gemini."""
import asyncio
import json
import logging
import warnings
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import config

logger = logging.getLogger("viral_clipper.llm")

class ClipJudgment(BaseModel):
    start: float = Field(..., description="Start timestamp of the clip in seconds")
    end: float = Field(..., description="End timestamp of the clip in seconds")
    duration: float = Field(..., description="Duration in seconds")
    title: str = Field(..., description="Catchy, high-CTR short title for the clip")
    hook_line: str = Field(..., description="The opening 3-second hook quote or phrase")
    score: int = Field(..., ge=0, le=100, description="Virality score from 0 to 100")
    viral_factors: List[str] = Field(default_factory=list, description="Keywords why it will perform, e.g. surprise, relatable, debate")
    reason: str = Field(..., description="1-2 sentences on why this clip will engage audiences")
    is_filler: bool = Field(False, description="Intro, outro, channel promo, or dead air: not a real moment")
    scene_type: str = Field(default="general", description="Category: fighting_action, comedy_humor, romantic_drama, or general")

def _init_gemini():
    if config.GEMINI_API_KEY:
        genai.configure(api_key=config.GEMINI_API_KEY)
    else:
        logger.warning("GEMINI_API_KEY not configured. LLM evaluations will use mock fallbacks.")

def _clean_json_str(text: str) -> str:
    """Strip markdown code fence blocks if returned."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()

# Keywords used for thematic heuristic classification
_FIGHT_WORDS = {
    "kill", "killed", "shoot", "shooting", "shot", "gun", "guns", "fight", "fighting",
    "war", "attack", "headshot", "clutch", "punch", "die", "dead", "destroy", "police",
    "chase", "enemy", "danger", "blast", "bomb", "hit", "sniper", "weapon", "rage",
    "maar", "maaro", "marna", "thok", "thoko", "goli", "goliyan", "dushman", "khatam",
    "bhaag", "bachao", "ladai", "lafda", "tod", "phod", "udaa", "loot", "gang", "gangster"
}
_COMEDY_WORDS = {
    "lol", "lmao", "rofl", "haha", "hahaha", "funny", "hilarious", "joke", "prank", "troll",
    "fail", "stupid", "crazy", "insane", "wtf", "roast", "comedy", "meme", "clown", "noob",
    "hasi", "has", "haso", "mazak", "pagal", "chutiya", "bewakoof", "gadha", "bakwaas", "popat",
    "hadd", "ulti", "maza", "arre", "bhains", "waah", "fatt", "halkat", "drama"
}
_ROMANCE_WORDS = {
    "love", "romance", "romantic", "flirt", "flirting", "cute", "crush", "dating", "marry",
    "propose", "girlfriend", "boyfriend", "gf", "bf", "kiss", "sweetheart", "darling", "baby",
    "beautiful", "heart", "crying", "miss", "emotional", "breakup", "sweet", "pyaar", "pyar",
    "ishq", "mohabbat", "dil", "jaan", "babu", "sundar", "khubsoorat", "shaadi", "yaad", "rona"
}

def _heuristic_fallback(
    candidate: Dict[str, Any],
    min_duration: float,
    max_duration: float,
    theme: str = "all"
) -> ClipJudgment:
    """Intelligent fallback when LLM is unavailable: detects Fighting, Comedy, Romance vs boring filler."""
    sig_score = int(candidate.get("signal_score", 0.5) * 100)
    dur = round(candidate.get("end", 0.0) - candidate.get("start", 0.0), 2)
    dur = max(min_duration, min(max_duration, dur))
    
    raw_text = candidate.get("text", "").strip()
    clean_text = raw_text.lower()
    tokens = set(re.findall(r"\b\w+\b", clean_text))
    
    fight_hits = len(tokens.intersection(_FIGHT_WORDS))
    comedy_hits = len(tokens.intersection(_COMEDY_WORDS))
    romance_hits = len(tokens.intersection(_ROMANCE_WORDS))

    # Detect exclamations and shouting
    has_exclamation = "!" in raw_text or any(w.isupper() and len(w) > 3 for w in raw_text.split())

    words = raw_text.split()
    sample_phrase = " ".join(words[:6]) if words else "Watch this moment"

    # Category determination
    if (theme in ("all", "fighting") and fight_hits >= 1) or (fight_hits > comedy_hits and fight_hits > romance_hits):
        scene_type = "fighting_action"
        final_score = min(98, max(75, sig_score + 25 + fight_hits * 5))
        title = f"{sample_phrase.upper()}! 🔥" if len(words) >= 3 else "INSANE FIGHT SCENE! 💥"
        hook = raw_text[:50] or "Intense battle underway!"
        viral_factors = ["fighting_action", "high_intensity", "combat_clutch"]
        reason = "Intense action/fighting sequence with explosive excitement and combat tension."
        is_filler = False
    elif (theme in ("all", "comedy") and comedy_hits >= 1) or (comedy_hits > fight_hits and comedy_hits > romance_hits):
        scene_type = "comedy_humor"
        final_score = min(98, max(75, sig_score + 25 + comedy_hits * 5))
        title = f"{sample_phrase.capitalize()} 😂" if len(words) >= 3 else "TRY NOT TO LAUGH! 🤣"
        hook = raw_text[:50] or "Wait for his funny reaction!"
        viral_factors = ["comedy_humor", "uncontrollable_laughter", "funny_fail"]
        reason = "Hilarious comedic scene featuring funny dialogue, trolling, and laugh reactions."
        is_filler = False
    elif (theme in ("all", "romance") and romance_hits >= 1) or (romance_hits > fight_hits and romance_hits > comedy_hits):
        scene_type = "romantic_drama"
        final_score = min(98, max(75, sig_score + 25 + romance_hits * 5))
        title = f"{sample_phrase.capitalize()} ❤️" if len(words) >= 3 else "WAIT FOR THIS MOMENT! 😳"
        hook = raw_text[:50] or "A very special interaction..."
        viral_factors = ["romantic_drama", "emotional_resonance", "relationship_banter"]
        reason = "Engaging romantic/dramatic scene with emotional connection and captivating banter."
        is_filler = False
    else:
        # Mundane / ambient / driving / random chatter
        scene_type = "general"
        final_score = min(45, max(15, sig_score - 20))
        title = " ".join(words[:7]) + ("..." if len(words) > 7 else "") if words else "Casual moment"
        hook = raw_text[:45] if raw_text else "Casual moment"
        viral_factors = ["general_ambient"]
        reason = "Casual ambient conversation without major action, comedy, or romantic payoff."
        is_filler = len(words) < 15 or not has_exclamation

    return ClipJudgment(
        start=candidate.get("start", 0.0),
        end=candidate.get("start", 0.0) + dur,
        duration=dur,
        title=title,
        hook_line=hook,
        score=final_score,
        viral_factors=viral_factors,
        reason=reason,
        is_filler=is_filler,
        scene_type=scene_type,
    )

_SAFETY = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


def _models_to_try() -> List[str]:
    models = [config.GEMINI_MODEL]
    if config.GEMINI_FALLBACK_MODEL and config.GEMINI_FALLBACK_MODEL not in models:
        models.append(config.GEMINI_FALLBACK_MODEL)
    return models


# When Google says a model's quota is used up (HTTP 429), stop calling that model for a while
# instead of waiting for timeouts on every job. {model_name: unix time it may be tried again}
_QUOTA_BLOCKED: Dict[str, float] = {}


def _is_quota_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "429" in msg or "quota" in msg or "resourceexhausted" in type(e).__name__.lower()


def gemini_available() -> bool:
    """False while every configured Gemini model is quota-blocked (callers can skip Gemini)."""
    import time
    return bool(config.GEMINI_API_KEY) and any(
        _QUOTA_BLOCKED.get(m, 0) <= time.time() for m in _models_to_try())


def _generate_json(prompt: str, timeout: float) -> Any:
    """Call Gemini (primary model, then fallback model) and parse a JSON response.
    Every request has a hard timeout: the SDK otherwise retries 5xx silently for ~10 minutes.
    Quota errors (429) are not retried, and that model is skipped for QUOTA_COOLDOWN_SEC."""
    import time
    from google.api_core import exceptions as gexc
    from google.api_core import retry as api_retry
    _init_gemini()
    # Retry only real hiccups (server busy / errors), never "quota exceeded"
    retry = api_retry.Retry(
        predicate=api_retry.if_exception_type(gexc.ServiceUnavailable, gexc.InternalServerError),
        initial=1.0, maximum=4.0, timeout=timeout,
    )
    last_err: Optional[Exception] = None
    for name in _models_to_try():
        blocked_until = _QUOTA_BLOCKED.get(name, 0)
        if blocked_until > time.time():
            last_err = RuntimeError(f"{name} quota exceeded (skipping for {int(blocked_until - time.time())}s)")
            continue
        try:
            model = genai.GenerativeModel(name, safety_settings=_SAFETY)
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"},
                request_options={"timeout": timeout, "retry": retry},
            )
            return json.loads(_clean_json_str(response.text))
        except Exception as e:
            last_err = e
            if _is_quota_error(e):
                _QUOTA_BLOCKED[name] = time.time() + config.QUOTA_COOLDOWN_SEC
                logger.warning(f"Gemini ({name}) quota exceeded; skipping it for "
                               f"{config.QUOTA_COOLDOWN_SEC // 60} min")
            else:
                logger.warning(f"Gemini ({name}) failed: {str(e)[:200]}")
    raise RuntimeError(f"All Gemini models failed: {last_err}")


def _clamp_judgment(data: Dict[str, Any], candidate: Dict[str, Any],
                    min_duration: float, max_duration: float, theme: str = "all") -> ClipJudgment:
    start = float(data.get("start", candidate.get("start", 0.0)))
    end = float(data.get("end", candidate.get("end", start + min_duration)))
    # Keep the LLM inside (roughly) the candidate window it was shown
    if not (candidate.get("start", 0.0) - 5 <= start <= candidate.get("end", start) + 5):
        start = float(candidate.get("start", 0.0))
        end = float(candidate.get("end", start + max_duration))
    duration = min(max_duration, max(min_duration, end - start))
    fb = _heuristic_fallback(candidate, min_duration, max_duration, theme=theme)
    return ClipJudgment(
        start=round(start, 2),
        end=round(start + duration, 2),
        duration=round(duration, 2),
        title=str(data.get("title") or fb.title),
        hook_line=str(data.get("hook_line") or fb.hook_line),
        score=int(max(0, min(100, int(data.get("score", fb.score))))),
        viral_factors=[str(v) for v in (data.get("viral_factors") or fb.viral_factors)][:6],
        reason=str(data.get("reason") or fb.reason),
        is_filler=bool(data.get("is_filler", fb.is_filler)),
        scene_type=str(data.get("scene_type") or fb.scene_type),
    )


def _title_language_rule(language: str) -> str:
    if config.CAPTION_SCRIPT == "roman" and language in ("hi", "ur", ""):
        return ('Write "title" and "hook_line" in HINGLISH: Hindi and English words mixed, written ONLY '
                'in Latin/English letters (e.g. "Bhai ye kya ho gaya", "Kal milte hain"). '
                'Never use Devanagari, and never fully translate into English.')
    return (f'Write "title" and "hook_line" in the SAME language and script as the transcript '
            f'(spoken language code: "{language or "unknown"}"). Do NOT translate to English.')


def _title_examples(language: str) -> str:
    if config.CAPTION_SCRIPT == "roman" and language in ("hi", "ur", ""):
        return '"BHAI YE KYA HO GAYA 😂", "SABSE KHATARNAK FIGHT 🔥", "WAIT FOR THIS MOMENT ❤️"'
    return '"HE DID NOT SURVIVE THIS 🔥", "TRY NOT TO LAUGH 😂", "THE MOST DRAMATIC MOMENT ❤️"'


def _batch_prompt(candidates: List[Dict[str, Any]], min_duration: float, max_duration: float,
                  language: str, theme: str = "all") -> str:
    blocks = []
    for i, c in enumerate(candidates):
        blocks.append(
            f"""### Candidate {i}
- Proposed Range: {c.get('start', 0.0):.2f}s to {c.get('end', 0.0):.2f}s
- Context Before: "{c.get('context_before', '')}"
- Clip Transcript: "{c.get('text', '')}"
- Context After: "{c.get('context_after', '')}"
- Raw Signal Score: {c.get('signal_score', 0.5):.2f}"""
        )

    focus_rules = ""
    if theme == "fighting":
        focus_rules = "- USER FOCUS: STRICTLY FIGHTING & ACTION SCENES. Look for combat, shootouts, clutch battles, attacks, and aggressive confrontations. Reject non-fighting clips."
    elif theme == "comedy":
        focus_rules = "- USER FOCUS: STRICTLY COMEDY & HUMOR SCENES. Look for laugh-out-loud moments, trolling, hilarious fails, funny roasts, and jokes. Reject non-funny clips."
    elif theme == "romance":
        focus_rules = "- USER FOCUS: STRICTLY ROMANTIC & DRAMA SCENES. Look for flirting, romantic banter, relationship drama, sweet moments, and emotional confessions."
    else:
        focus_rules = "- CORE FOCUS: You MUST extract the best scenes featuring FIGHTING & ACTION (combat, clutches, shootouts, high tension), COMEDY & HUMOR (jokes, laughter, trolling, hilarious fails), or ROMANTIC & DRAMA (flirting, cute banter, emotional moments)."

    return f"""You are an elite short-form video editor for TikTok, YouTube Shorts, and Instagram Reels.
Below are {len(candidates)} candidate moments from one video/stream. Judge EACH one to extract high-engagement, viral clips.

CRITICAL THEMATIC RULES:
{focus_rules}
1. Give TOP VIRAL SCORES (80-99) to genuine FIGHTING/ACTION, COMEDY/HUMOR, or ROMANTIC/DRAMA scenes.
2. HEAVILY PENALIZE RANDOM FILLER (Score below 35, set "is_filler": true) for boring driving, idle walking, menu navigation, silent gameplay, or mundane repetitive banter with no action, comedy, or romance payoff. NEVER return a boring random clip.
3. Clip duration MUST be between {min_duration:.1f} and {max_duration:.1f} seconds, inside or near the proposed range.
4. Reward a strong hook in the first 3 seconds (gunshot, scream, punchline, surprise, controversy, big reaction, conflict).
5. The clip must make sense on its own and end cleanly (punchline, laughter, conclusion, cliffhanger).
6. Classify each clip's "scene_type": "fighting_action" | "comedy_humor" | "romantic_drama" | "general".
7. {_title_language_rule(language)}
   Titles: short (3-7 words), catchy, curiosity-driven, scroll-stopping, highlighting the fighting, comedy, or romance (e.g. {_title_examples(language)}); 1 relevant emoji.

{chr(10).join(blocks)}

Return STRICT JSON: an array with exactly one object per candidate:
[{{"id": int (candidate number), "start": float, "end": float, "title": str, "hook_line": str,
   "score": int, "viral_factors": [str], "reason": str, "is_filler": bool, "scene_type": str}}]
"""


def judge_candidate(
    candidate: Dict[str, Any],
    min_duration: float = config.DEFAULT_CLIP_MIN_DURATION,
    max_duration: float = config.DEFAULT_CLIP_MAX_DURATION,
    language: str = "",
    theme: str = "all",
) -> ClipJudgment:
    """Judge a single candidate (blocking). Falls back to a heuristic pick on any failure."""
    if not config.GEMINI_API_KEY:
        return _heuristic_fallback(candidate, min_duration, max_duration, theme=theme)
    try:
        data = _generate_json(_batch_prompt([candidate], min_duration, max_duration, language, theme=theme),
                              config.LLM_TIMEOUT_SEC)
        if isinstance(data, list):
            data = data[0] if data else {}
        return _clamp_judgment(data, candidate, min_duration, max_duration, theme=theme)
    except Exception:
        return _heuristic_fallback(candidate, min_duration, max_duration, theme=theme)


async def judge_candidates(
    candidates: List[Dict[str, Any]],
    min_duration: float = config.DEFAULT_CLIP_MIN_DURATION,
    max_duration: float = config.DEFAULT_CLIP_MAX_DURATION,
    language: str = "",
    theme: str = "all",
) -> List[ClipJudgment]:
    """Judge all candidates in ONE Gemini request (fast, rate-limit friendly, and the model can
    compare moments against each other). Any candidate the model skips gets a heuristic pick."""
    if not candidates:
        return []
    results: Dict[int, ClipJudgment] = {}
    if config.GEMINI_API_KEY:
        prompt = _batch_prompt(candidates, min_duration, max_duration, language, theme=theme)
        try:
            data = await asyncio.to_thread(_generate_json, prompt, config.LLM_TIMEOUT_SEC)
            if isinstance(data, dict):
                data = data.get("candidates") or data.get("clips") or [data]
            for pos, item in enumerate(data if isinstance(data, list) else []):
                if not isinstance(item, dict):
                    continue
                idx = int(item.get("id", pos)) if str(item.get("id", pos)).isdigit() else pos
                if 0 <= idx < len(candidates) and idx not in results:
                    results[idx] = _clamp_judgment(item, candidates[idx], min_duration, max_duration, theme=theme)
            logger.info(f"Gemini judged {len(results)}/{len(candidates)} candidates in one request")
        except Exception as e:
            logger.warning(f"Batched Gemini judging failed, using heuristic picks: {e}")
    else:
        logger.warning("GEMINI_API_KEY not set; using heuristic picks.")

    judgments = [results.get(i) or _heuristic_fallback(c, min_duration, max_duration, theme=theme)
                 for i, c in enumerate(candidates)]
    judgments.sort(key=lambda j: j.score, reverse=True)
    return judgments
