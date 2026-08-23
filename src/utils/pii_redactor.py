"""PII redaction and prompt-injection detection for VitalSense.

Both functions are intentionally isolated from the rest of the app so they
can be unit-tested before being wired into the retrieval/analysis pipeline.

redact_pii(text)              — masks identifiable patient fields in-place.
detect_injection_attempt(text) — returns True if text contains known injection
                                  patterns; caller decides how to handle it.
"""
import re
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII redaction rules
# Each tuple is (field_label, compiled_regex, replacement_string).
# Patterns are intentionally anchored so they only match labelled fields,
# not arbitrary occurrences of the same words in clinical text.
# ---------------------------------------------------------------------------

_PII_RULES: list[tuple[str, re.Pattern, str]] = [
    # Patient name:  "Patient: John Doe" / "Patient Name: John Doe"
    # Captures everything after the colon until a pipe, newline, or end-of-line.
    (
        "PATIENT_NAME",
        re.compile(
            r"(Patient\s*(?:Name)?\s*:\s*)([^\n|]+)",
            re.IGNORECASE,
        ),
        r"\1[REDACTED_NAME]",
    ),
    # Date of birth: "DOB: 01/01/1980" / "Date of Birth: 1 Jan 1980"
    (
        "DATE_OF_BIRTH",
        re.compile(
            r"((?:Date\s+of\s+Birth|DOB)\s*:\s*)([^\n|]+)",
            re.IGNORECASE,
        ),
        r"\1[REDACTED_DOB]",
    ),
    # Age (labelled): "Age: 42" — short numeric value after the label
    # We only redact when explicitly labelled to avoid clobbering clinical values.
    (
        "AGE",
        re.compile(
            r"((?<![a-z])Age\s*:\s*)(\d{1,3}(?:\s*(?:years?|yrs?|y\.?o\.?))?)",
            re.IGNORECASE,
        ),
        r"\1[REDACTED_AGE]",
    ),
    # Patient / Medical Record ID / MRN
    (
        "PATIENT_ID",
        re.compile(
            r"((?:Patient\s+ID|MRN|Medical\s+Record\s+(?:Number|No\.?))\s*:\s*)([^\n|]+)",
            re.IGNORECASE,
        ),
        r"\1[REDACTED_ID]",
    ),
    # Phone number (basic international + US formats)
    (
        "PHONE",
        re.compile(
            r"((?:Phone|Tel|Mobile|Contact)\s*(?:No\.?|Number)?\s*:\s*)"
            r"(\+?[\d\s\-().]{7,20})",
            re.IGNORECASE,
        ),
        r"\1[REDACTED_PHONE]",
    ),
    # Email address (labelled or bare)
    (
        "EMAIL",
        re.compile(
            r"([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z]{2,})",
        ),
        r"[REDACTED_EMAIL]",
    ),
]


def redact_pii(text: str) -> str:
    """Return *text* with identifiable patient fields replaced by safe tokens.

    Detection is regex-based (no ML). Each rule targets a labelled field, so
    the risk of clobbering clinical values (e.g. numeric lab results) is low.

    Logs the *type* of each redaction at DEBUG level for auditability — the
    actual PII value is never logged.

    Args:
        text: Raw report text as extracted from the PDF.

    Returns:
        Redacted copy of *text*; the original string is never modified.
    """
    redacted = text
    for field_label, pattern, replacement in _PII_RULES:
        new_text, n = pattern.subn(replacement, redacted)
        if n:
            logger.debug("PII redacted: field=%s occurrences=%d", field_label, n)
        redacted = new_text
    return redacted


# ---------------------------------------------------------------------------
# Prompt injection detection
# ---------------------------------------------------------------------------

# Patterns are checked case-insensitively against the full input text.
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"\bact\s+as\b", re.IGNORECASE),
    re.compile(r"\bpretend\s+(you\s+are|to\s+be)\b", re.IGNORECASE),
    re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE),
    re.compile(r"\breveal\s+your\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\s+follow\b", re.IGNORECASE),
    re.compile(r"\bnew\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bDAN\b"),  # "Do Anything Now" jailbreak keyword
]


def detect_injection_attempt(text: str) -> bool:
    """Return True if *text* contains a known prompt-injection pattern.

    Checks are case-insensitive regex matches. Returns on the *first* match
    so it is fast even for large documents.

    Args:
        text: Text to scan (PDF content or user chat message).

    Returns:
        True  — injection pattern detected; caller should refuse/warn.
        False — no known pattern found.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning(
                "Prompt injection pattern detected: %r matched in text (first 200 chars): %r",
                pattern.pattern,
                text[:200],
            )
            return True
    return False


def strip_injection_lines(text: str) -> str:
    """Remove individual lines from *text* that match injection patterns.

    Designed for PDF report content where the document as a whole should NOT
    be rejected (unlike a user query) — only the offending lines are stripped
    so that legitimate clinical content is preserved for analysis.

    Uses line-level granularity (split on ``\\n``) which matches the
    line-oriented structure of lab reports and avoids sentence-splitting risk.

    Logs at WARNING level the *count* of removed lines for auditability;
    the actual line content is never logged.

    Args:
        text: Report text (post PII redaction).

    Returns:
        Copy of *text* with injection lines removed. Unchanged if clean.
    """
    lines = text.split("\n")
    clean_lines: list[str] = []
    removed = 0

    for line in lines:
        matched = any(pattern.search(line) for pattern in _INJECTION_PATTERNS)
        if matched:
            removed += 1
        else:
            clean_lines.append(line)

    if removed:
        logger.warning(
            "Injection lines stripped from uploaded document: count=%d", removed
        )

    return "\n".join(clean_lines)

