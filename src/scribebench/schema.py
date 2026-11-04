from copy import deepcopy
from dataclasses import asdict, dataclass, field

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
    if parse.confidence < threshold or parse.intent not in INTENTS:
        return state, "clarify"
    slots = parse.slots
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
