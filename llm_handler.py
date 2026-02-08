"""LLM handler for the requirements analysis assistant.

This module provides functions to:
- load a prompt template from disk
- combine preprocessed sentences into a single input string
- insert the input text into the prompt template
- send the prompt to an LLM (OpenAI supported) with graceful error handling
- parse the LLM response into a structured Python list/dict

The module is dependency-light and provides a `mock` mode so the
project can be demonstrated without API keys during a viva.
"""

from typing import List, Dict, Any, Optional
import os
import json
import re
import logging

logger = logging.getLogger(__name__)


class LLMHandlerError(Exception):
    pass


def load_prompt(path: str, encoding: str = "utf-8") -> str:
    """Load a prompt template from `path` and return its contents.

    The template can include a placeholder `{text}` where the combined
    sentences will be inserted. If no placeholder is present, the
    combined text is appended to the end of the template.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding=encoding) as fh:
        return fh.read()


def combine_sentences(sentences: List[str], separator: str = "\n") -> str:
    """Combine preprocessed sentences into a single string for the LLM.

    A simple join is used so the original sentence boundaries are
    preserved for analysis by the LLM.
    """
    return separator.join(s.strip() for s in sentences if s and s.strip())


def insert_into_prompt(template: str, text: str, placeholder: str = "{text}") -> str:
    """Insert `text` into `template` at `placeholder` or append if absent.

    Returns the finalized prompt string.
    """
    if placeholder in template:
        return template.replace(placeholder, text)
    # fallback: append with a separator
    return template.rstrip() + "\n\n" + text


def _extract_json_substring(text: str) -> Optional[str]:
    """Attempt to find a JSON array or object substring inside `text`.

    Returns the substring or None.
    """
    # try array first
    arr_match = re.search(r"(\[\s*\{.*\}\s*\])", text, re.S)
    if arr_match:
        return arr_match.group(1)
    obj_match = re.search(r"(\{.*\})", text, re.S)
    if obj_match:
        return obj_match.group(1)
    return None


def parse_llm_output(output: str) -> List[Dict[str, Any]]:
    """Parse the LLM `output` into a structured list of requirement dicts.

    Expected format (preferred): JSON list of objects with keys:
    - `text`
    - `classification` ('Functional'|'Non-Functional')
    - `subtype` (optional)
    - `explanation` (optional)

    The parser first tries to load JSON directly; if that fails it
    extracts a JSON-looking substring, then falls back to a line-based
    parser. The function raises `LLMHandlerError` only if parsing fails
    or if the result cannot be coerced into the expected structure.
    """
    if not output or not output.strip():
        return []

    # Attempt direct JSON parse
    try:
        data = json.loads(output)
        if isinstance(data, list):
            return data
        # if it's a dict with a top-level items list, try to extract
        if isinstance(data, dict):
            # find the first list-valued field that looks right
            for v in data.values():
                if isinstance(v, list):
                    return v
            # fallback: wrap dict
            return [data]
    except Exception:
        pass

    # Try to extract a JSON substring
    js = _extract_json_substring(output)
    if js:
        try:
            data = json.loads(js)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
        except Exception:
            logger.debug("Failed to parse extracted JSON substring", exc_info=True)

    # Fallback: line-based parsing. Expect lines like:
    # Requirement: <text> | Classification: <...> | Subtype: <...> | Explanation: <...>
    items: List[Dict[str, Any]] = []
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    for ln in lines:
        # try to split by '|' first
        parts = [p.strip() for p in ln.split("|") if p.strip()]
        entry: Dict[str, Any] = {}
        # if there's only one part, treat it as free-form text
        if len(parts) == 1:
            entry["text"] = parts[0]
            items.append(entry)
            continue
        for p in parts:
            # split on ':' first occurrence
            if ":" in p:
                k, v = p.split(":", 1)
                key = k.strip().lower()
                val = v.strip()
                if "class" in key:
                    entry["classification"] = val
                elif "subtype" in key:
                    entry["subtype"] = val
                elif "explain" in key:
                    entry["explanation"] = val
                else:
                    # treat remaining as text if not set
                    if "text" not in entry:
                        entry["text"] = val
                    else:
                        # append to explanation or text
                        entry.setdefault("explanation", "")
                        entry["explanation"] += " " + val
            else:
                # no colon - if text not set, set it
                if "text" not in entry:
                    entry["text"] = p
                else:
                    entry.setdefault("explanation", "")
                    entry["explanation"] += " " + p
        if entry:
            items.append(entry)

    if not items:
        raise LLMHandlerError("Failed to parse LLM output into structured items")

    return items


def call_openai(prompt: str, model: str = "gpt-4", max_tokens: int = 1500, timeout: int = 30) -> str:
    """Call OpenAI's API and return the model text response.

    Requires the `openai` package and an `OPENAI_API_KEY` environment
    variable. Errors are captured and raised as `LLMHandlerError`.
    """
    try:
        import openai
    except Exception as exc:
        raise LLMHandlerError("OpenAI package not available") from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise LLMHandlerError("OPENAI_API_KEY not set in environment")
    openai.api_key = api_key

    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            timeout=timeout,
        )
        # extract text from response
        choices = resp.get("choices")
        if choices and len(choices) > 0:
            content = choices[0].get("message", {}).get("content") or choices[0].get("text")
            return content or ""
        return ""
    except Exception as exc:
        logger.exception("OpenAI API call failed")
        raise LLMHandlerError("OpenAI API call failed") from exc


def call_llm(
    prompt: str,
    provider: str = "openai",
    model: str = "gpt-4",
    mock: bool = False,
    **kwargs,
) -> str:
    """Send `prompt` to the chosen LLM provider and return raw text.

    If `mock` is True the function returns a synthetic response using
    local heuristics (no external API call). This is useful for demos
    and vivas where API keys may not be available.
    """
    if mock:
        # produce a simple JSON output using local classifier+explainability
        try:
            from . import classifier, explainability

            # naive: assume the prompt ends with the combined text
            # take last 2000 chars to find sentences
            body = prompt[-4000:]
            # split by newlines and sentences
            sentences = [ln.strip() for ln in body.splitlines() if ln.strip()][:200]
            classified = classifier.classify_requirements(sentences)
            explained = explainability.explain_requirements(classified)
            return json.dumps(explained, indent=2)
        except Exception:
            # last resort: echo the prompt
            return json.dumps([{"text": "(mock) " + prompt[:200]}])

    if provider.lower() == "openai":
        return call_openai(prompt, model=model, **kwargs)

    raise LLMHandlerError(f"Unsupported provider: {provider}")


def generate_requirements_from_llm(
    sentences: List[str],
    prompt_template: str,
    provider: str = "openai",
    model: str = "gpt-4",
    mock: bool = False,
) -> List[Dict[str, Any]]:
    """High-level helper: combine sentences, build prompt, call LLM and parse output.

    Returns a list of requirement dictionaries with at least the `text`
    and `classification` fields. Raises `LLMHandlerError` for fatal
    failures.
    """
    combined = combine_sentences(sentences)
    prompt = insert_into_prompt(prompt_template, combined)

    raw = call_llm(prompt, provider=provider, model=model, mock=mock)

    parsed = parse_llm_output(raw)
    # ensure each parsed item contains at least `text` and `classification`
    normalized: List[Dict[str, Any]] = []
    for it in parsed:
        if not isinstance(it, dict):
            continue
        text = it.get("text") or it.get("requirement") or None
        classification = it.get("classification") or it.get("type") or None
        subtype = it.get("subtype") or it.get("category") or None
        explanation = it.get("explanation") or it.get("reason") or None
        normalized.append(
            {
                "text": text,
                "classification": classification,
                "subtype": subtype,
                "explanation": explanation,
            }
        )

    return normalized


if __name__ == "__main__":
    # Demo runner: use `mock=True` so this file can be run in CI/viva.
    sample_sentences = [
        "The system shall allow users to register and login using email and password.",
        "Responses to user queries must be returned within 2 seconds under normal load.",
        "All sensitive data must be encrypted at rest and in transit.",
        "The UI should be intuitive and accessible to screen readers.",
    ]

    # Try to load a prompt from prompts/requirement_prompt.txt if present
    prompt_path = os.path.join(os.getcwd(), "prompts", "requirement_prompt.txt")
    if os.path.exists(prompt_path):
        template = load_prompt(prompt_path)
    else:
        template = "Classify the following requirements into Functional or Non-Functional.\n\n{text}\n\nReturn a JSON array of objects with fields: text, classification, subtype (optional), explanation (optional)."

    out = generate_requirements_from_llm(sample_sentences, template, mock=True)
    print(json.dumps(out, indent=2))
