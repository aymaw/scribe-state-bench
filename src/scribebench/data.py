"""Fictional medications and transcript-like text; no audio or patient records."""

import hashlib
import json
import random
import re

from .schema import INTENTS, TAGS, Parse, State, transition

DRUGS = ["medorin", "calmexa", "norava", "zelorin", "luma plus", "nova forte"]
SYMPTOMS = ["headache", "nausea", "blurred vision", "dry eyes", "dizziness", "sore throat"]
TEMPLATES = {
    "add_medication": [
        "i take {drug}",
        "my medication is {drug} {dose} mg {frequency}",
        "i use {drug} {dose} mg {frequency}",
        "currently taking {drug} {dose} mg {frequency}",
        "put {drug} {dose} mg {frequency} on my list",
        "um i am on {drug} {dose} mg {frequency}",
    ],
    "stop_medication": [
        "i stopped {drug}",
        "remove {drug}",
        "i no longer take {drug}",
        "discontinue {drug} from my list",
        "i quit using {drug}",
        "uh {drug} is no longer a medication i take",
    ],
    "correct_dose": [
        "actually {dose} mg",
        "change the dose to {dose} mg",
        "make that {dose} mg",
        "the correct amount is {dose} mg",
        "i meant {dose} mg instead",
        "sorry it should read {dose} mg",
    ],
    "unknown_dose": [
        "i do not know the dose",
        "the dose is unknown",
        "i forgot the dose",
        "i cannot recall the amount",
        "leave the dose unspecified",
        "um i am unsure how much i take",
    ],
    "report_symptom": [
        "i have {symptom}",
        "i am experiencing {symptom}",
        "there is {symptom}",
        "i notice {symptom}",
        "lately i have had {symptom}",
        "um the problem is {symptom}",
    ],
    "deny_symptom": [
        "i do not have {symptom}",
        "no {symptom}",
        "i deny {symptom}",
        "i am not experiencing {symptom}",
        "there has been no {symptom}",
        "uh i have not noticed {symptom}",
    ],
    "confirm": [
        "yes that is correct",
        "confirmed",
        "that looks right",
        "everything listed is accurate",
        "you have it right",
        "um yes exactly",
    ],
}


def tokens(text):
    # Match original text so Unicode case expansion cannot shift character offsets.
    return list(re.finditer(r"[a-zA-Z0-9]+", text))


def render(intent, family, values):
    template = TEMPLATES[intent][family]
    used = re.findall(r"\{(\w+)\}", template)
    text = template.format(**values)
    spans, tags = [], ["O"] * len(tokens(text))
    for key in used:
        value = values[key]
        start = text.index(value)
        end = start + len(value)
        spans.append({"slot": key, "start": start, "end": end, "value": value})
        positions = [i for i, m in enumerate(tokens(text)) if m.start() >= start and m.end() <= end]
        for j, pos in enumerate(positions):
            tags[pos] = ("B-" if j == 0 else "I-") + key
    return {
        "text": text,
        "intent": intent,
        "slots": {k: values[k] for k in used},
        "spans": spans,
        "tags": tags,
        "family": f"{intent}/{family}",
    }


def generate(n=600, seed=52):
    if n < 30:
        raise ValueError("n must be at least 30")
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    for i in range(n):
        split = "train" if i < int(0.6 * n) else "validation" if i < int(0.8 * n) else "test"
        choices = [0, 1, 2] if split == "train" else [3] if split == "validation" else [4, 5]
        values = {
            "drug": rng.choice(DRUGS),
            "dose": str(rng.choice([5, 10, 20, 40])),
            "frequency": rng.choice(["daily", "nightly", "twice daily"]),
            "symptom": rng.choice(SYMPTOMS),
        }
        sequence = [
            "add_medication",
            "correct_dose",
            "unknown_dose",
            "report_symptom",
            "deny_symptom",
            "confirm",
            "stop_medication",
        ]
        # Different surface-family partition throughout a dialogue; split at dialogue level.
        state, turns = State(), []
        for intent in sequence:
            if intent == "correct_dose":
                values["dose"] = str(rng.choice([5, 10, 20, 40]))
            turn = render(intent, rng.choice(choices), values)
            assert all(tag in TAGS for tag in turn["tags"])
            assert turn["intent"] in INTENTS
            state, _ = transition(state, Parse(intent, turn["slots"]))
            turn["state"] = state.snapshot()
            turns.append(turn)
        splits[split].append({"id": f"synthetic-{i:05d}", "turns": turns})
    return splits


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
