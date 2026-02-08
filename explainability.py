"""Explainability helpers for classified requirements.

This module accepts requirement dictionaries with the keys:
- `text`: requirement text
- `classification`: 'Functional' or 'Non-Functional'
- `subtype`: optional subtype string for Non-Functional requirements

It adds an `explanation` field to each requirement describing why the
classification was made. The module is intentionally simple and
rule-based so explanations are transparent for a final-year project viva.
"""

from typing import List, Dict, Optional
import re

from . import classifier

_WORD_RE = re.compile(r"\b[a-zA-Z0-9\-]+\b")


def _tokenize(text: str) -> List[str]:
    """Return lowercased tokens from `text`.

    Matches alphanumeric and simple hyphenated tokens, consistent with
    the tokenization used in `classifier`.
    """
    return [t.lower() for t in _WORD_RE.findall(text)]


# Local keyword sets mirror the classifier's categories to produce
# coherent, human-readable explanations.
_FUNCTIONAL_KEYWORDS = set(classifier._FUNCTIONAL_KEYWORDS) if hasattr(classifier, "_FUNCTIONAL_KEYWORDS") else set()
_NONFUNCTIONAL_KEYWORDS = set(classifier._NONFUNCTIONAL_KEYWORDS) if hasattr(classifier, "_NONFUNCTIONAL_KEYWORDS") else set()

_SUBTYPE_KEYWORDS = {
    "Security": {
        "secure",
        "security",
        "encrypt",
        "authentication",
        "authorization",
        "confidentiality",
        "integrity",
        "privacy",
        "vulnerability",
        "password",
        "token",
        "ssl",
        "tls",
    },
    "Performance": {
        "performance",
        "latency",
        "throughput",
        "scalability",
        "response",
        "time",
        "seconds",
        "ms",
        "milliseconds",
        "fast",
        "slow",
        "uptime",
    },
    "Usability": {
        "usability",
        "user-friendly",
        "user",
        "ux",
        "ui",
        "intuitive",
        "learnability",
        "accessible",
        "accessibility",
        "easy",
    },
    "Reliability": {
        "reliability",
        "available",
        "availability",
        "backup",
        "failover",
        "redundancy",
        "recover",
        "fault",
        "robust",
    },
}


def explain_requirement(req: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """Return the requirement dict augmented with an `explanation` field.

    Expects `req` to contain `text` and `classification`. If
    `classification` is 'Non-Functional', the function will also consult
    the optional `subtype` value.

    The explanation cites matching keywords or the scoring rationale so
    it is easy to explain during a viva.
    """
    if not isinstance(req, dict):
        raise TypeError("requirement must be a dict with keys 'text' and 'classification'")

    text = (req.get("text") or "").strip()
    classification = req.get("classification") or "Unknown"
    subtype = req.get("subtype")

    tokens = set(_tokenize(text))

    f_matches = sorted(tokens & _FUNCTIONAL_KEYWORDS)
    nf_matches = sorted(tokens & _NONFUNCTIONAL_KEYWORDS)

    # If subtype is present but empty string, normalize to None
    if subtype == "":
        subtype = None

    explanation = ""

    if classification == "Functional":
        if f_matches:
            explanation = (
                "Classified as Functional because it contains action keywords: "
                + ", ".join(f_matches)
                + "."
            )
        else:
            # fallback to scoring evidence
            f_score = sum(1 for t in tokens if t in _FUNCTIONAL_KEYWORDS)
            nf_score = sum(1 for t in tokens if t in _NONFUNCTIONAL_KEYWORDS)
            explanation = (
                f"Classified as Functional by scoring (functional={f_score}, nonfunctional={nf_score})."
            )

    elif classification == "Non-Functional":
        # Prefer subtype-specific explanation when available
        chosen_subtype = subtype
        if not chosen_subtype:
            # ask classifier for a consistent subtype if possible
            try:
                chosen_subtype = classifier.detect_nonfunctional_subtype(text)
            except Exception:
                chosen_subtype = None

        if chosen_subtype:
            subtype_kw = _SUBTYPE_KEYWORDS.get(chosen_subtype, set())
            matched = sorted(tokens & subtype_kw)
            if matched:
                explanation = (
                    f"Classified as Non-Functional ({chosen_subtype}) because it mentions: "
                    + ", ".join(matched)
                    + "."
                )
            else:
                # no direct subtype matches, but classifier decided it's that subtype
                explanation = (
                    f"Classified as Non-Functional ({chosen_subtype}). Keyword evidence is indirect; "
                    f"detected by subtype heuristics."
                )
        else:
            if nf_matches:
                explanation = (
                    "Classified as Non-Functional because it contains non-functional keywords: "
                    + ", ".join(nf_matches)
                    + "."
                )
            else:
                # fallback to scoring evidence
                f_score = sum(1 for t in tokens if t in _FUNCTIONAL_KEYWORDS)
                nf_score = sum(1 for t in tokens if t in _NONFUNCTIONAL_KEYWORDS)
                explanation = (
                    f"Classified as Non-Functional by scoring (functional={f_score}, nonfunctional={nf_score})."
                )

    else:
        explanation = "Classification unavailable; cannot generate explanation."

    req_with_expl = dict(req)
    req_with_expl["explanation"] = explanation
    return req_with_expl


def explain_requirements(requirements: List[Dict[str, Optional[str]]]) -> List[Dict[str, Optional[str]]]:
    """Add explanations for a list of requirement dictionaries.

    Returns a new list where each item has an added `explanation` field.
    """
    results: List[Dict[str, Optional[str]]] = []
    for r in requirements:
        results.append(explain_requirement(r))
    return results


if __name__ == "__main__":
    # Demo: classify some sample requirements using `classifier`, then
    # generate explanations.
    samples = [
        "The system shall allow users to register and login using email and password.",
        "Responses to user queries must be returned within 2 seconds under normal load.",
        "All sensitive data must be encrypted at rest and in transit.",
        "The UI should be intuitive and accessible to screen readers.",
        "The application shall generate monthly financial reports.",
    ]

    classified = classifier.classify_requirements(samples)
    explained = explain_requirements(classified)

    for item in explained:
        print(item)
