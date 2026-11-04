import math
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from numbers import Real

INTENTS = [
    "add_medication",
    "stop_medication",
    "correct_dose",
    "unknown_dose",
    "report_symptom",
    "deny_symptom",
    "confirm",
]
SLOTS = ["drug", "dose", "frequency", "symptom"]
TAGS = ["O"] + [prefix + "-" + slot for slot in SLOTS for prefix in ("B", "I")]


@dataclass
class Parse:
    intent: str
    slots: dict[str, str]
    confidence: float = 1.0


@dataclass
class State:
    medications: dict = field(default_factory=dict)
    symptoms: dict = field(default_factory=dict)
    focus: str | None = None

    def snapshot(self):
        return deepcopy(asdict(self))


def transition(state, parse, threshold=0.55):
    """Pure reducer. Unknown details stay null. No clinical inference or external effects."""
    state = deepcopy(state)
    allowed = {
        "add_medication": {"drug", "dose", "frequency"},
        "stop_medication": {"drug"},
        "correct_dose": {"drug", "dose"},
        "unknown_dose": {"drug"},
        "report_symptom": {"symptom"},
        "deny_symptom": {"symptom"},
        "confirm": set(),
    }
    if (
        not isinstance(parse.confidence, Real)
        or not math.isfinite(parse.confidence)
        or not 0 <= parse.confidence <= 1
        or parse.confidence < threshold
        or parse.intent not in INTENTS
        or not isinstance(parse.slots, dict)
        or set(parse.slots) - allowed[parse.intent]
        or any(
            not isinstance(v, str) or not v.strip() or len(v) > 128 for v in parse.slots.values()
        )
    ):
        return state, "clarify"
    slots = {key: value.strip().lower() for key, value in parse.slots.items()}
    if "dose" in slots and (
        not re.fullmatch(r"\d+(?:\.\d+)?", slots["dose"])
        or not 0 < float(slots["dose"]) < float("inf")
    ):
        return state, "clarify"
    if "frequency" in slots and slots["frequency"] not in {"daily", "nightly", "twice daily"}:
        return state, "clarify"
    intent = parse.intent
    drug = slots.get("drug")
    if intent == "add_medication":
        if not drug:
            return state, "clarify"
        record = state.medications.setdefault(drug, {"dose": None, "frequency": None})
        for name in ("dose", "frequency"):
            if name in slots:
                record[name] = slots[name]
        state.focus = drug
    elif intent == "stop_medication":
        if not drug or drug not in state.medications:
            return state, "clarify"
        del state.medications[drug]
        if state.focus == drug:
            state.focus = None
    elif intent in ("correct_dose", "unknown_dose"):
        target = drug or state.focus
        if target not in state.medications or (intent == "correct_dose" and not slots.get("dose")):
            return state, "clarify"
        state.medications[target]["dose"] = slots["dose"] if intent == "correct_dose" else None
        state.focus = target
    elif intent in ("report_symptom", "deny_symptom"):
        if "symptom" not in slots:
            return state, "clarify"
        state.symptoms[slots["symptom"]] = "present" if intent == "report_symptom" else "denied"
    return state, "accepted"
