"""Classifier module for requirement type identification.

This module provides simple, rule-based classification of extracted
requirements into `Functional` or `Non-Functional`. For Non-Functional
requirements it can optionally identify common subtypes such as
`Performance`, `Security`, `Usability`, and `Reliability`.

The implementation is intentionally lightweight and dependency-free so
it can be easily integrated with preprocessing, an LLM handler, and
explainability modules in the project.

Functions
- classify_requirement(text, detect_subtype=True)
- classify_requirements(requirements, detect_subtypes=True)
- detect_nonfunctional_subtype(text)
"""

from typing import List, Dict, Optional
import re

__all__ = [
	"classify_requirement",
	"classify_requirements",
	"detect_nonfunctional_subtype",
	"tokenize",
]


_WORD_RE = re.compile(r"\b[a-zA-Z0-9\-]+\b")

# Basic keyword sets. These can be extended or replaced by ML models later.
_FUNCTIONAL_KEYWORDS = {
	"create",
	"read",
	"update",
	"delete",
	"generate",
	"display",
	"show",
	"send",
	"receive",
	"process",
	"validate",
	"store",
	"retrieve",
	"search",
	"sort",
	"export",
	"import",
	"login",
	"logout",
	"register",
	"submit",
	"approve",
	"notify",
	"report",
	"schedule",
	"allow",
	"enable",
	"manage",
	"configure",
	"connect",
	"upload",
	"download",
	"filter",
	"select",
	"support",
	"provide",
	"shall",
}

_NONFUNCTIONAL_KEYWORDS = {
	"performance",
	"latency",
	"throughput",
	"scalability",
	"secure",
	"security",
	"encrypt",
	"authentication",
	"authorization",
	"availability",
	"reliability",
	"usability",
	"user-friendly",
	"user",
	"accessibility",
	"response",
	"time",
	"fast",
	"slow",
	"seconds",
	"ms",
	"milliseconds",
	"uptime",
	"backup",
	"failover",
	"redundancy",
	"privacy",
	"integrity",
	"confidentiality",
	"vulnerability",
	"recover",
	"robust",
	"maintainability",
}


def _tokenize(text: str) -> List[str]:
	"""Return lowercased word tokens from text.

	Keeps alphanumeric tokens and simple hyphenated words.
	"""
	return [t.lower() for t in _WORD_RE.findall(text)]


def tokenize(text: str) -> List[str]:
	"""Public tokenizer that returns lowercased tokens.

	This wraps the module-internal `_tokenize` so other modules can
	reuse a single tokenization strategy (avoids duplicate logic).
	"""
	return _tokenize(text)


def detect_nonfunctional_subtype(text: str) -> Optional[str]:
	"""Identify a likely Non-Functional subtype from the requirement.

	Returns one of: 'Performance', 'Security', 'Usability', 'Reliability',
	or `None` if no subtype is confidently detected.

	The function uses keyword matching; ordering reflects a preference
	for more specific, easy-to-detect categories first.
	"""
	tokens = set(_tokenize(text))

	security = {
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
	}

	performance = {
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
	}

	usability = {
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
	}

	reliability = {
		"reliability",
		"available",
		"availability",
		"backup",
		"failover",
		"redundancy",
		"recover",
		"fault",
		"robust",
	}

	if tokens & security:
		return "Security"
	if tokens & performance:
		return "Performance"
	if tokens & usability:
		return "Usability"
	if tokens & reliability:
		return "Reliability"

	return None


def classify_requirement(text: str, detect_subtype: bool = True) -> Dict[str, Optional[str]]:
	"""Classify a single requirement as Functional or Non-Functional.

	Returns a dictionary with keys:
	- `text`: the original requirement string
	- `classification`: 'Functional' or 'Non-Functional'
	- `subtype`: subtype string if Non-Functional and `detect_subtype` is True

	The decision uses simple keyword scoring: functional and
	non-functional keyword occurrences are counted and compared. If
	non-functional keywords dominate, the requirement is labelled
	'Non-Functional'. Ties favor 'Functional' (conservative).
	"""
	tokens = _tokenize(text)
	f_score = sum(1 for t in tokens if t in _FUNCTIONAL_KEYWORDS)
	nf_score = sum(1 for t in tokens if t in _NONFUNCTIONAL_KEYWORDS)

	classification = "Functional" if f_score >= nf_score else "Non-Functional"
	subtype = None
	if classification == "Non-Functional" and detect_subtype:
		subtype = detect_nonfunctional_subtype(text)

	return {"text": text, "classification": classification, "subtype": subtype}


def classify_requirements(requirements: List[str], detect_subtypes: bool = True) -> List[Dict[str, Optional[str]]]:
	"""Classify a list of requirement strings.

	`detect_subtypes` toggles whether subtype identification is attempted
	for Non-Functional requirements.
	"""
	results: List[Dict[str, Optional[str]]] = []
	for r in requirements:
		results.append(classify_requirement(r, detect_subtype=detect_subtypes))
	return results


if __name__ == "__main__":
	# Small demo when module is run directly.
	sample_requirements = [
		"The system shall allow users to register and login using email and password.",
		"Responses to user queries must be returned within 2 seconds under normal load.",
		"All sensitive data must be encrypted at rest and in transit.",
		"The UI should be intuitive and accessible to screen readers.",
		"The application shall generate monthly financial reports.",
	]

	classified = classify_requirements(sample_requirements)
	for item in classified:
		print(item)