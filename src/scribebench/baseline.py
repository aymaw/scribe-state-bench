import re

from .data import DRUGS, SYMPTOMS
from .schema import Parse


def parse_rule(text):
    source = text
    text = text.lower()
    spans = []
    for key, choices in [("drug", DRUGS), ("symptom", SYMPTOMS)]:
        for value in choices:
            for match in re.finditer(r"\b" + re.escape(value) + r"\b", source, re.IGNORECASE):
                spans.append(
                    {"slot": key, "start": match.start(), "end": match.end(), "value": value}
                )
    for key, pattern in [
        ("dose", r"\b\d+(?= mg\b)"),
        ("frequency", r"\b(?:twice daily|daily|nightly)\b"),
    ]:
        for match in re.finditer(pattern, source, re.IGNORECASE):
            spans.append(
                {
                    "slot": key,
                    "start": match.start(),
                    "end": match.end(),
                    "value": match.group().lower(),
                }
            )
    slots = {s["slot"]: s["value"] for s in spans}
    if len(slots) != len(spans):
        return Parse("unsupported", {}, 0.0), spans
    if "drug" in slots:
        remainder = text.replace("no longer", "")
        if re.search(r"\b(no|not|never|without)\b|n't", remainder):
            return Parse("unsupported", {}, 0.0), spans
    if "symptom" in slots:
        intent = "deny_symptom" if re.search(r"\b(no|not|deny)\b", text) else "report_symptom"
    elif re.search(r"\b(stopped|remove|longer|discontinue|quit)\b", text):
        intent = "stop_medication"
    elif re.search(r"unknown|forgot|unsure|unspecified|cannot recall|do not know", text):
        intent = "unknown_dose"
    elif "dose" in slots and "drug" not in slots:
        intent = "correct_dose"
    elif "drug" in slots:
        intent = "add_medication"
    elif re.search(r"correct|confirmed|looks right|accurate|have it right|exactly", text):
        intent = "confirm"
    else:
        return Parse("unsupported", {}, 0.0), []
    return Parse(intent, slots), spans
