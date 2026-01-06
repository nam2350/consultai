"""Utility helpers for post-processing Korean consultation summaries across models."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

LABEL_ALIASES: Dict[str, Tuple[str, ...]] = {
    "\uace0\uac1d": ("\uace0\uac1d", "\uc774\uc6a9\uc790", "\uc0ac\uc6a9\uc790", "\ubbfc\uc6d0\uc778"),
    "\uc0c1\ub2f4\uc0ac": ("\uc0c1\ub2f4\uc0ac", "\uc0c1\ub2f4\uc6d0", "\ub2f4\ub2f9\uc790", "\uc9c1\uc6d0"),
    "\uc0c1\ub2f4\uacb0\uacfc": ("\uc0c1\ub2f4\uacb0\uacfc", "\ucc98\ub9ac\uacb0\uacfc", "\uacb0\uacfc", "\uc870\uce58\uacb0\uacfc", "\ucd5c\uc885\uacb0\uacfc"),
}

_PLACEHOLDER = "\ub0b4\uc6a9\uc774 \ucda9\ubd84\ud558\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4."

def _remove_prompt_artifacts(text: str, patterns: Iterable[str]) -> str:
    """Remove prompt or system guidance fragments from the raw summary."""
    cleaned = text
    lowered = cleaned.lower()
    for pattern in patterns:
        if not pattern:
            continue
        target = pattern.lower()
        idx = lowered.find(target)
        if idx != -1:
            fragment = cleaned[idx:].split("\n", 1)
            cleaned = fragment[1].strip() if len(fragment) > 1 else fragment[0].strip()
            lowered = cleaned.lower()
    return cleaned


def _normalize_inline_whitespace(text: str) -> str:
    """Normalize inline whitespace and convert escaped newline tokens."""
    if not text:
        return ""
    normalized = text.replace("\\r", " ").replace("\\n", " ")
    normalized = re.sub(r"\s{2,}", " ", normalized)
    return normalized.strip()




def _strip_label_prefix(line: str, aliases: Iterable[str]) -> str:
    """Strip known section labels (and common variants) from a line."""
    candidate = _normalize_inline_whitespace(line)
    if not candidate:
        return ""

    for alias in aliases:
        patterns = (
            re.compile(r"^\s*\*{0,2}" + re.escape(alias) + r"\*{0,2}\s*[:：]\s*", re.IGNORECASE),
            re.compile(r"^\s*" + re.escape(alias) + r"\s*[:：]\s*", re.IGNORECASE),
        )
        stripped = candidate
        for pattern in patterns:
            if pattern.match(stripped):
                stripped = pattern.sub("", stripped, count=1)
                stripped = _normalize_inline_whitespace(stripped)
                break
        if stripped != candidate:
            return stripped
    return candidate


def _collect_section_contents(lines: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """Extract section contents for ??/???/???? with graceful fallbacks."""
    remaining = [line.strip() for line in lines if line.strip()]
    contents: Dict[str, str] = {}

    for label in ("\uace0\uac1d", "\uc0c1\ub2f4\uc0ac", "\uc0c1\ub2f4\uacb0\uacfc"):
        aliases = LABEL_ALIASES[label]
        content = ""

        for idx, line in enumerate(remaining):
            stripped = _strip_label_prefix(line, aliases)
            if stripped != line:
                content = stripped
                remaining.pop(idx)
                break
            if any(alias in line for alias in aliases):
                content = stripped
                remaining.pop(idx)
                break

        if not content and remaining:
            fallback_line = remaining.pop(0)
            content = _strip_label_prefix(fallback_line, aliases)

        cleaned_content = _normalize_inline_whitespace(content)

        placeholder_token = cleaned_content.replace("-", "").strip() if cleaned_content else ""
        if not placeholder_token:
            cleaned_content = ""

        contents[label] = cleaned_content if cleaned_content else _PLACEHOLDER

    return contents, remaining


def build_three_section_summary(
    lines: List[str],
    *,
    min_length: int,
    logger: Optional[logging.Logger] = None,
    fallback_text: str = "",
) -> str:
    """Create a three-line consultation summary with robust fallbacks."""
    contents, remaining = _collect_section_contents(lines)

    summary_lines = [
        f"**{label}**: {contents[label]}"
        for label in ("\uace0\uac1d", "\uc0c1\ub2f4\uc0ac", "\uc0c1\ub2f4\uacb0\uacfc")
    ]
    summary = "\n".join(summary_lines)

    if len(summary) < min_length:
        extras = [line for line in remaining if line.strip()]
        if not extras and fallback_text:
            sanitized_fallback = _prepare_summary_text(fallback_text)
            extras = [
                segment.strip()
                for segment in re.split(r"(?<=[.!?\?])\s+", sanitized_fallback)
                if segment.strip()
            ]

        filtered_extras: List[str] = []
        for extra in extras:
            extra_stripped = extra.strip()
            if not extra_stripped:
                continue
            if re.fullmatch(r"-{3,}", extra_stripped):
                continue
            if re.search(r"\*\*[^*]+\*\*\s*[:\uff1a]", extra_stripped):
                continue
            filtered_extras.append(extra_stripped)
        extras = filtered_extras

        result_key = "\uc0c1\ub2f4\uacb0\uacfc"
        result_content = contents[result_key]
        for extra in extras:
            cleaned = _strip_label_prefix(extra, LABEL_ALIASES[result_key])
            if not cleaned or cleaned in contents.values():
                continue
            result_content = f"{result_content} {cleaned}".strip()
            contents[result_key] = result_content
            summary_lines[-1] = f"**{result_key}**: {result_content}"
            summary = "\n".join(summary_lines)
            if len(summary) >= min_length:
                break

        if len(summary) < min_length and logger:
            logger.warning(
                "Summary shorter than expected (len=%d, min=%d)",
                len(summary),
                min_length,
            )

    return summary



def _prepare_summary_text(text: str) -> str:
    """Normalize raw summary text by removing separators and enforcing section breaks."""
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace(" ", "\n").replace(" ", "\n")
    cleaned = cleaned.replace("\\r", "\n").replace("\\n", "\n")
    cleaned = re.sub(r"(?m)^\s*-{3,}\s*$", "\n", cleaned)
    cleaned = re.sub(r"(?<!^)(?<!\n)(\*\*[^*]{1,24}\*\*\s*[:\uff1a])", r"\n\1", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()


def normalize_summary(
    raw_summary: str,
    *,
    cleanup_patterns: Iterable[str] = (),
    min_length: int,
    logger: Optional[logging.Logger] = None,
    fallback_text: Optional[str] = None,
) -> str:
    """Normalize a raw model response into the standard three-line summary format."""
    if not raw_summary or not raw_summary.strip():
        raise RuntimeError("Generated summary is empty or too short")

    normalized = raw_summary.replace("\r\n", "\n").strip()
    if cleanup_patterns:
        normalized = _remove_prompt_artifacts(normalized, cleanup_patterns)

    normalized = _prepare_summary_text(normalized)
    if not normalized:
        raise RuntimeError("Generated summary is empty after normalization")

    lines = [line.strip() for line in normalized.split("\n") if line.strip() or line == ""]
    fallback_source = _prepare_summary_text(fallback_text) if fallback_text else normalized
    if not fallback_source:
        fallback_source = normalized

    return build_three_section_summary(
        lines,
        min_length=min_length,
        logger=logger,
        fallback_text=fallback_source,
    )


import unicodedata


def _h(*names: str) -> str:
    return ''.join(unicodedata.lookup(f"HANGUL SYLLABLE {name}") for name in names)


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\uAC00-\uD7A3][A-Za-z0-9\uAC00-\uD7A3+/_.-]*")
_KOREAN_STOPWORDS = {
    _h('GO', 'GAEG'),
    _h('GO', 'GAEG', 'NIM'),
    _h('SANG', 'DAM'),
    _h('SANG', 'DAM', 'SA'),
    _h('SANG', 'DAM', 'WEON'),
    _h('MUN', 'YI'),
    _h('AN', 'NAE'),
    _h('GAM', 'SA'),
    _h('YO', 'CEONG'),
    _h('CEO', 'RI'),
    _h('HWAG', 'IN'),
    _h('SA', 'YONG'),
    _h('GWAN', 'RYEON'),
    _h('JEONG', 'BO'),
    _h('HYEON', 'JAE'),
    _h('NAE', 'YONG'),
    _h('I', 'HU'),
    _h('CU', 'HU'),
    _h('GYEONG', 'U'),
    _h('BU', 'TAK'),
    _h('JI', 'WEON'),
    _h('JEONG', 'MAL'),
    _h('DDAE', 'MUN'),
    _h('YO', 'CEONG', 'SA', 'HANG'),
    _h('I', 'JJOG'),
    _h('GEU', 'JJOG'),
    _h('JEO', 'JJOG'),
    _h('YEO', 'GI'),
    _h('GEO', 'GI'),
    _h('I', 'GEO'),
    _h('GEU', 'GEO'),
    _h('JEO', 'GEO'),
    _h('GEU', 'REOM'),
    _h('GEU', 'REO', 'MYEON'),
    _h('SANG', 'TAE'),
    _h('MUN', 'JE'),
    _h('AN', 'NYEONG', 'HA', 'SE', 'YO'),
    _h('AN', 'NYEONG', 'HA', 'SIB', 'NI', 'GGA')
}
_KOREAN_STOPWORDS.update({
    '그리고',
    '근데',
    '그런데',
    '그러면',
    '그러니까',
    '그래서',
    '그래가지고',
    '그다음',
    '그다음으로',
    '다음으로',
    '이어서',
    '그러고',
    '그렇다면',
})

_ENGLISH_STOPWORDS = {
    'customer',
    'agent',
    'thanks',
    'thank',
    'please',
    'request',
    'issue',
    'problem',
    'information',
    'details',
    'detail',
    'hello',
    'hi',
    'regards',
    'regarding',
    'about',
    'support',
    'help',
    'today',
    'now',
    'currently',
    'later',
    'again',
    'team',
    'manager',
    'department',
    'section',
    'case',
    'status',
    'process',
    'processing',
    'confirm',
    'confirmation',
    'provide',
    'provided',
    'using',
    'use',
    'related',
    'etc'
}
_KOREAN_SUFFIXES = tuple(sorted((
    _h('IP', 'NI', 'DA'),
    _h('IP', 'NI', 'DA', 'MAN'),
    _h('IP', 'NI', 'DA', 'YO'),
    _h('SEUB', 'NI', 'DA'),
    _h('SEUB', 'NI', 'GGA'),
    _h('HAB', 'NI', 'DA'),
    _h('HAB', 'NI', 'DA', 'MAN'),
    _h('HAB', 'NI', 'DA', 'YO'),
    _h('HA', 'SE', 'YO'),
    _h('DEU', 'RIB', 'NI', 'DA'),
    _h('DEU', 'RIL', 'GE', 'YO'),
    _h('EUN'),
    _h('NEUN'),
    _h('I'),
    _h('GA'),
    _h('EUL'),
    _h('REUL'),
    _h('DO'),
    _h('MAN'),
    _h('WA'),
    _h('GWA'),
    _h('EU', 'RO'),
    _h('EU', 'RO', 'NEUN'),
    _h('EU', 'RO', 'MAN'),
    _h('EU', 'RO', 'SEO'),
    _h('EU', 'RO', 'SEO', 'DO'),
    _h('EU', 'RO', 'SEO', 'SE'),
    _h('EU', 'RO', 'SEO', 'YO'),
    _h('E'),
    _h('E', 'SEO'),
    _h('E', 'SEO', 'YO')
), key=len, reverse=True))
_STOPWORD_SUBSTRINGS = (
    _h('GAM', 'SA', 'HAB'),
    _h('AN', 'NYEONG'),
    _h('BU', 'TAK', 'DEU'),
    _h('DO', 'WA', 'DEU'),
    _h('HWAG', 'IN', 'DEU'),
    _h('AN', 'NAE', 'DEU'),
    _h('JEONG', 'BO', 'DEU'),
    _h('SEUB', 'NI')
)

_STOPWORD_SUBSTRINGS += (
    "\uC5B4\uB5A0",
    "\uC5B4\uB5A0\uAC8C",
    "\uC624\uB978",
    "\uC788\uC5B4",
    "\uAC74\uAC00",
    "\uAC74\uAC00\uC694",
    "\uC5EC\uAE30\uC11C",
    "\uC5EC\uAE30\uB294",
    "\uADFC\uB370",
    "\uADF8\uB2E4\uC74C",
    "\uADF8\uB2E4\uC74C\uC73C\uB85C",
    "\uB2E4\uC74C\uC73C\uB85C",
    "\uC774\uC5B4\uC11C",
)
_ALLOWED_SHORT_ENGLISH = {
    'kt',
    'ai',
    'ivr',
    'otp',
    'sms',
    'api'
}
_GREETING_PREFIXES = (
    _h('AN', 'NYEONG'),
    _h('GAM', 'SA'),
    _h('JOE', 'SONG')
)


def _strip_korean_suffixes(token: str) -> str:
    """Remove common particles and polite endings from a Korean token."""
    base = token
    changed = True
    while changed:
        changed = False
        for suffix in _KOREAN_SUFFIXES:
            if len(base) <= len(suffix):
                continue
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                changed = True
                break
    return base


def _normalize_keyword_candidate(token: str) -> Tuple[str, str]:
    """Normalize a raw token and return (display, normalized) pair."""
    cleaned = unicodedata.normalize("NFKC", token)
    strip_chars = "_-`~!@#$%^&*()+={}[]|\\:;<>?,./\"'"
    cleaned = cleaned.strip(strip_chars)
    cleaned = cleaned.replace("\u2022", "").replace("\u00b7", "")
    if not cleaned:
        return "", ""

    if cleaned.isdigit():
        return "", ""

    has_korean = any('\uAC00' <= ch <= '\uD7A3' for ch in cleaned)

    if has_korean:
        stripped = _strip_korean_suffixes(cleaned)
        stripped = stripped.strip()
        if len(stripped) <= 1:
            stripped = cleaned.strip()
        normalized = stripped.lower()
        display = stripped
    else:
        normalized = cleaned.lower()
        display = cleaned.upper() if cleaned.isalpha() and len(cleaned) <= 4 else cleaned

    if len(normalized) <= 1:
        return "", ""

    if normalized in _KOREAN_STOPWORDS or normalized in _ENGLISH_STOPWORDS:
        return "", ""

    if any(part in normalized for part in _STOPWORD_SUBSTRINGS):
        return "", ""

    if has_korean:
        conversational_prefixes = ("\uC5B4\uB5A0", "\uC5B4\uB514", "\uC5B4\uB290")
        if any(normalized.startswith(prefix) for prefix in conversational_prefixes):
            return "", ""
        conversational_suffixes = ("\uAC00", "\uAC00\uC694", "\uAC00\uC2DC", "\uAC8C", "\uAC8C\uC694", "\uC694", "\uB098\uC694", "\uB2C8\uAE4C")
        if any(normalized.endswith(suffix) for suffix in conversational_suffixes):
            return "", ""


    if len(normalized) <= 2 and normalized not in _ALLOWED_SHORT_ENGLISH:
        return "", ""

    return display, normalized


def extract_consultation_keywords(text: str, max_keywords: int = 3) -> List[str]:
    """Extract domain-specific keywords from consultation text."""
    if not text:
        return []

    counter: Counter[str] = Counter()
    display_map: Dict[str, str] = {}

    for token in _TOKEN_PATTERN.findall(text):
        display, normalized = _normalize_keyword_candidate(token)
        if not normalized:
            continue
        counter[normalized] += 1
        display_map.setdefault(normalized, display)

    if not counter:
        return []

    ranked = sorted(counter.items(), key=lambda item: (-item[1], -len(item[0])))

    keywords: List[str] = []
    for normalized, _ in ranked:
        keywords.append(display_map.get(normalized, normalized))
        if len(keywords) >= max_keywords:
            break

    return keywords


def build_keyword_title_from_text(text: str, max_terms: int = 3) -> str:
    """Construct an underscore-joined keyword title from consultation text."""
    keywords = extract_consultation_keywords(text, max_terms)
    if not keywords:
        return ""
    return "_".join(keywords[:max_terms])


def extract_descriptive_title_candidate(text: str, max_length: int = 28) -> str:
    """Pick a concise descriptive sentence from the consultation text."""
    if not text:
        return ""

    for raw_sentence in re.split(r"[\n.!?]", text):
        candidate = raw_sentence.strip()
        if not candidate:
            continue
        candidate = re.sub(r"^[^:\uFF1A]+[:\uFF1A]\s*", "", candidate)
        if len(candidate) < 6:
            continue
        lowered = unicodedata.normalize("NFKC", candidate).lower()
        if any(lowered.startswith(prefix) for prefix in _GREETING_PREFIXES):
            continue
        if any(part in lowered for part in ('\uAC10\uC0AC', '\uC548\uB155')):
            continue
        if len(candidate) > max_length:
            candidate = candidate[: max_length - 3].rstrip() + "..."
        return candidate

    return ""
